"""Phase 8 research-grid runner tests (`docs/plan.md` Phase 8).

Concrete-ML has no Windows wheels, so every test here is `skipif`-gated,
matching every other Phase 5/6/7 FHE test, and built against tiny
SYNTHETIC LR/XGBoost models saved to `tmp_path` -- never the real
committed Phase 5/6/8 data. Verifies: the full `run_configuration` schema
(accuracy + latency + provenance), the n_bits override actually changes
what gets compiled, checkpoint-based resumability across two calls, and
that a deliberately-failing accuracy gate is reported, not hidden.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.provenance import compute_config_hash
from src.fhe.handoff import extract_lr_pipeline_params, save_handoff


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_concrete_ml_available(),
    reason="concrete-ml is not installed in this environment. It has no Windows wheels; run under the WSL2 venv. See docs/environment.md.",
)


def _make_lr_source_config(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(1)
    n, d = 500, 4
    X = rng.randn(n, d) * np.array([1.0, 50.0, 0.01, 200.0])
    y = (X[:, 0] + X[:, 1] / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(X, y)
    params = extract_lr_pipeline_params(pipeline)
    reference_prob = pipeline.predict_proba(X)[:, 1]

    # RAW X, not scaler-transformed -- matches src/fhe/export.py's real
    # convention exactly: `rebuild_pipeline_predict_proba` applies the
    # scaler itself from `params`, so storing already-scaled X here would
    # double-scale and silently corrupt T0 (this was caught by T0 actually
    # failing during development, not assumed).
    params_path = tmp_path / "lr_params.json"
    params_path.write_text(json.dumps(params))
    npz_path = tmp_path / "lr_handoff.npz"
    manifest_path = tmp_path / "lr_handoff_manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X.astype(np.float64), X_val=X.astype(np.float64), y_val=y, val_transaction_ids=np.arange(n),
        reference_val_prob=reference_prob, columns=[f"f{i}" for i in range(d)],
        manifest_extra={"params_path": str(params_path), "threshold": 0.5},
    )
    source_config = {"n_bits": 8, "output": {"handoff_npz": str(npz_path), "handoff_manifest": str(manifest_path)}}
    source_config_path = tmp_path / "fake_lr_source.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config))
    return source_config_path, source_config


def _base_grid_config(tmp_path, entries):
    return {
        "seed": 42,
        "latency_trials": 2,
        "correctness_sample": {"n": 2, "n_positive": 1},
        "row_selection": {"n_positive": 0},
        "disable_chunk_size": 1000,
        "t2": {"rows": 50, "chunk_size": 50},
        "tolerances": {"t0_max_abs_diff": 1e-6, "t3_min_decision_agreement": 0.5, "t3_max_pr_auc_drop": 1.0},
        "output": {"dir": str(tmp_path / "results"), "handoff_dir": str(tmp_path / "handoff_ckpt")},
        "configurations": entries,
    }


def _make_xgb_source_config(tmp_path):
    import xgboost as xgb

    from src.fhe.export_xgboost import inference_booster
    from src.fhe.handoff import sha256_file

    rng = np.random.RandomState(2)
    n, d = 600, 4
    X = np.column_stack([rng.rand(n) * 100, rng.rand(n), rng.randn(n), rng.rand(n) * 5])
    y = ((X[:, 0] / 100 + X[:, 1] + 0.3 * rng.rand(n)) > 1.1).astype(int)
    X_train, y_train, X_val, y_val = X[:400], y[:400], X[400:], y[400:]

    model = xgb.XGBClassifier(n_estimators=40, max_depth=2, learning_rate=0.5, early_stopping_rounds=5, eval_metric="aucpr", random_state=0, n_jobs=1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    sliced, n_stored, best_iteration = inference_booster(model)
    booster_path = tmp_path / "xgb.json"
    sliced.save_model(booster_path)
    reference_prob = model.predict_proba(X_val)[:, 1]

    npz_path = tmp_path / "xgb_handoff.npz"
    manifest_path = tmp_path / "xgb_handoff_manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X_train, X_val=X_val, y_val=y_val, val_transaction_ids=np.arange(len(X_val)),
        reference_val_prob=reference_prob, columns=[f"f{i}" for i in range(d)],
        manifest_extra={"booster_path": str(booster_path), "booster_sha256": sha256_file(booster_path), "n_trees_inference": best_iteration + 1, "threshold": 0.5},
    )
    source_config = {
        "n_bits": 6, "seed": 42,
        "calibration": {"rows": "all", "include_extremes": False},
        "output": {"handoff_dir": str(tmp_path), "dir": str(tmp_path / "results_placeholder")},
    }
    # _load_xgboost reads {output.dir}/{tier}/handoff_manifest.json and {output.handoff_dir}/xgb_{tier}.npz
    (tmp_path / "results_placeholder" / "top_toy").mkdir(parents=True)
    manifest_path.rename(tmp_path / "results_placeholder" / "top_toy" / "handoff_manifest.json")
    npz_path.rename(tmp_path / f"xgb_top_toy.npz")
    source_config_path = tmp_path / "fake_xgb_source.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config))
    return source_config_path, source_config


def test_run_configuration_end_to_end_on_synthetic_xgboost(tmp_path):
    """Same schema/gate contract as the LR test, exercised through
    `_accuracy_xgboost`'s different (integer-level T1/T2) code path."""
    from src.benchmark.phase8_grid import run_configuration

    source_config_path, _ = _make_xgb_source_config(tmp_path)
    entry = {"label": "xgb_synth_bits6", "model_type": "xgboost", "tier": "top_toy", "n_bits": 6, "source_config": str(source_config_path)}
    config = _base_grid_config(tmp_path, [entry])

    result = run_configuration(config, "fake_grid.yaml", "xgb_synth_bits6")

    assert result["status"] == "passed", result.get("error")
    assert result["n_bits"] == 6
    assert result["fhe_latency"]["total"]["n_trials"] == 2
    assert len(result["accuracy"]["correctness_sample_positions"]) == 2
    for gate in ("t0", "t3", "t2", "t1_correctness"):
        assert result["accuracy"]["gates"][gate]["passed"] is True


def test_run_configuration_end_to_end_on_synthetic_lr_and_resumes_via_checkpoint(tmp_path):
    from src.benchmark.phase8_grid import run_configuration

    source_config_path, source_config = _make_lr_source_config(tmp_path)
    entry = {"label": "lr_synth_bits8", "model_type": "lr", "n_bits": 8, "source_config": str(source_config_path)}
    config = _base_grid_config(tmp_path, [entry])
    # Tight T3: with the compiled LR consuming standardized inputs (the scaler-omission
    # fix), quantization alone must stay well inside the project's real-data bars' spirit.
    config["tolerances"].update({"t3_min_decision_agreement": 0.9, "t3_max_pr_auc_drop": 0.1})

    result = run_configuration(config, "fake_grid.yaml", "lr_synth_bits8")

    assert result["status"] == "passed", result.get("error") or result["gates_passed"]
    assert result["gates_passed"]["t3"] is True
    assert result["n_bits"] == 8
    assert result["compile_seconds"] > 0
    assert result["source_config_hash"] == compute_config_hash(source_config)
    assert result["test_partition_touched"] is False

    # Latency: 5-repeated-executions-of-one-row shape (here 2, per config)
    assert result["plaintext_latency"]["n_trials"] == 2
    assert result["fhe_latency"]["total"]["n_trials"] == 2
    assert "position" in result["row_selection"] and "seed" in result["row_selection"]

    # Accuracy: correctness sample structurally separate from latency trials
    assert len(result["accuracy"]["correctness_sample_positions"]) == 2
    for gate in ("t0", "t3", "t2", "t1_correctness"):
        assert "passed" in result["accuracy"]["gates"][gate]
    for metric_block in ("quantized_full_metrics", "float_full_metrics"):
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2"):
            assert key in result["accuracy"][metric_block]

    assert result["all_gates_passed"] is True

    # Resumability: re-running the SAME label must not re-execute the FHE
    # trials (checkpoints reused) and must reach the identical status.
    result2 = run_configuration(config, "fake_grid.yaml", "lr_synth_bits8")
    assert result2["status"] == "passed"
    assert result2["fhe_latency"]["total"]["trials_seconds"] == result["fhe_latency"]["total"]["trials_seconds"]


def test_n_bits_override_actually_changes_what_is_compiled(tmp_path):
    """Regression lock for the one thing Phase 8 adds to Phase 7's loaders:
    two entries pointing at the SAME source config but different `n_bits`
    must produce a DIFFERENTLY quantized model -- not silently both
    compiling at the source config's own n_bits=8. Checked directly at the
    quantizer level (guaranteed to differ), not via an emergent circuit
    statistic that might coincidentally match for a tiny toy circuit."""
    from src.benchmark.phase8_grid import _load_configuration

    source_config_path, _ = _make_lr_source_config(tmp_path)
    entry8 = {"label": "lr_bits8", "model_type": "lr", "n_bits": 8, "source_config": str(source_config_path)}
    entry12 = {"label": "lr_bits12", "model_type": "lr", "n_bits": 12, "source_config": str(source_config_path)}

    loaded8 = _load_configuration(entry8)
    loaded12 = _load_configuration(entry12)

    assert loaded8["n_bits"] == 8
    assert loaded12["n_bits"] == 12
    q8 = loaded8["cml_model"].quantize_input(loaded8["X_val"][:20])
    q12 = loaded12["cml_model"].quantize_input(loaded12["X_val"][:20])
    assert not np.array_equal(q8, q12)  # same 20 rows, different n_bits -> different quantization


def test_failed_accuracy_gate_is_reported_not_hidden(tmp_path):
    """A configuration whose handoff's reference probabilities are
    deliberately wrong (T0 cannot pass) must be reported as
    `failed_accuracy_gates` with the specific gate identified, never
    silently passed or silently dropped from the summary."""
    from src.benchmark.phase8_grid import run_configuration

    source_config_path, source_config = _make_lr_source_config(tmp_path)
    manifest_path = Path(source_config["output"]["handoff_manifest"])
    npz_path = Path(source_config["output"]["handoff_npz"])
    manifest = json.loads(manifest_path.read_text())
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = dict(npz.items())
    arrays["reference_val_prob"] = arrays["reference_val_prob"].copy()
    arrays["reference_val_prob"][:] = 1.0 - arrays["reference_val_prob"]  # deliberately, unmistakably wrong
    save_handoff(
        npz_path, manifest_path,
        X_train=arrays["X_train"], X_val=arrays["X_val"], y_val=arrays["y_val"],
        val_transaction_ids=arrays["val_transaction_ids"], reference_val_prob=arrays["reference_val_prob"],
        columns=manifest["columns"], manifest_extra={k: v for k, v in manifest.items() if k not in ("npz_sha256", "columns", "n_train", "n_val", "n_features")},
    )

    entry = {"label": "lr_impossible_t0", "model_type": "lr", "n_bits": 8, "source_config": str(source_config_path)}
    config = _base_grid_config(tmp_path, [entry])

    result = run_configuration(config, "fake_grid.yaml", "lr_impossible_t0")

    assert result["status"] == "failed_accuracy_gates"
    assert result["all_gates_passed"] is False
    assert result["gates_passed"]["t0"] is False
