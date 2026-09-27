"""Phase 9 MLP grid runner tests (`docs/plan.md` Phase 9).

`skipif`-gated on Concrete-ML (WSL2 FHE venv only) and built on a tiny SYNTHETIC tier written to `tmp_path` -- never the
real Phase 9 data. Verifies the full `run_configuration` schema and gate results (including one REAL encrypt->run->decrypt),
that the reported latency statistics equal the raw trials, checkpoint reuse and invalidation, that the float twin is shared
across bit-widths, and that an infeasible or failing configuration is RECORDED with its reason instead of raised or dropped.
"""

import importlib.util
import json

import numpy as np
import pytest
import yaml

from src.fhe.handoff import save_handoff, sha256_file


def _concrete_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _concrete_available(),
    reason="concrete-ml is not installed in this environment. It has no Windows wheels; run under the WSL2 venv. See docs/environment.md.",
)

TIER = "toy"
MLP_CFG = {"n_layers": 3, "n_hidden_neurons_multiplier": 2, "n_accum_bits": 16, "lr": 0.01, "batch_size": 128, "max_epochs": 3, "early_stopping_patience": 2, "min_delta": 0.0}


@pytest.fixture
def grid(tmp_path):
    from src.data.provenance import compute_config_hash  # noqa: F401  (imported to fail early if provenance moves)

    rng = np.random.default_rng(0)
    d, n_train, n_val = 12, 1500, 600
    w = rng.normal(size=d)
    X = rng.normal(size=(n_train + n_val, d)) * np.linspace(1, 5, d)
    y = ((X @ w / 5 + rng.normal(scale=0.6, size=n_train + n_val)) > 1.0).astype(np.int64)
    X_train, y_train, X_val, y_val = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    assert y_val.sum() > 5 and y_train.sum() > 20

    out_dir, handoff_dir = tmp_path / "results", tmp_path / "handoff"
    scaler = {"scaler_mean": X_train.mean(axis=0).tolist(), "scaler_scale": X_train.std(axis=0).tolist()}
    scaler_path = tmp_path / "scaler.json"
    scaler_path.write_text(json.dumps(scaler))
    export_dir = out_dir / "export" / TIER
    export_dir.mkdir(parents=True)
    save_handoff(
        handoff_dir / f"mlp_{TIER}.npz", export_dir / "handoff_manifest.json",
        X_train=X_train, X_val=X_val, y_val=y_val, val_transaction_ids=np.arange(n_val), reference_val_prob=None,
        columns=[f"f{i}" for i in range(d)], y_train=y_train,
        manifest_extra={"tier": TIER, "scaler_path": str(scaler_path), "scaler_sha256": sha256_file(scaler_path)},
    )
    config = {
        "seed": 42, "latency_trials": 2, "correctness_sample": {"n": 2, "n_positive": 1}, "row_selection": {"n_positive": 0},
        "mlp": dict(MLP_CFG), "compile_calibration": {"rows": 200, "include_extremes": True}, "disable_chunk_size": 500,
        "t2": {"rows": 30, "chunk_size": 30}, "tolerances": {"t3_min_decision_agreement": 0.0, "t3_max_pr_auc_drop": 1.0},
        "seed_stability": {"seeds": [42, 43]}, "max_key_material_gb": 50.0,
        "output": {"dir": str(out_dir), "handoff_dir": str(handoff_dir)},
        "configurations": [{"label": "mlp_toy_bits4", "tier": TIER, "n_bits": 4}, {"label": "mlp_toy_bits3", "tier": TIER, "n_bits": 3}],
    }
    config_path = tmp_path / "grid.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config, str(config_path), out_dir


@pytest.fixture
def completed(grid):
    from src.benchmark import phase9_grid

    config, config_path, out_dir = grid
    result = phase9_grid.run_configuration(config, config_path, "mlp_toy_bits4")
    return config, config_path, out_dir, result


def test_full_schema_and_correctness_gates_on_a_toy_tier(completed):
    _, _, out_dir, result = completed
    assert result["status"] in ("passed", "failed_accuracy_gates") and result["model_type"] == "mlp" and result["test_partition_touched"] is False
    gates = result["accuracy"]["gates"]
    assert set(gates) == {"t0", "t3", "t2", "t1_correctness"}
    assert gates["t0"]["passed"] and gates["t0"]["max_abs_diff"] == 0.0  # reload is exact
    assert gates["t2"]["passed"] and gates["t2"]["n_rows"] == 30  # simulate == clear integers, exactly
    assert gates["t1_correctness"]["passed"] and gates["t1_correctness"]["decrypted_vs_disable_exact_integer_match"]  # real FHE, exact
    assert result["circuit_stats"]["programmable_bootstrap_count"] > 0 and result["circuit_stats"]["max_integer_bit_width"] <= 16
    assert result["training"]["qat"]["epochs_run"] >= 1 and result["compile_seconds"] > 0
    assert result["ciphertext_size_bytes"]["input"] > 0 and result["key_size_bytes"]["bootstrap"] > 0
    assert (out_dir / "mlp_toy_bits4" / "metrics.json").exists() and (out_dir / "mlp_toy_bits4" / "provenance.json").exists()


def test_reported_latency_statistics_equal_the_raw_trials(completed):
    from src.analysis.phase8_results import verify_means_against_raw

    _, _, _, result = completed
    assert result["fhe_latency"]["total"]["n_trials"] == 2 and result["fhe_latency"]["outputs_reproducible"] is True
    assert verify_means_against_raw(result, expected_trials=2) == []


def test_checkpoints_are_reused_on_a_second_run_with_identical_fhe_trials(completed):
    from src.benchmark import phase9_grid

    config, config_path, _, first = completed
    second = phase9_grid.run_configuration(config, config_path, "mlp_toy_bits4")
    assert second["training"]["qat_checkpoint_reused"] is True and second["training"]["float_twin_checkpoint_reused"] is True
    assert second["fhe_latency"]["total"]["trials_seconds"] == first["fhe_latency"]["total"]["trials_seconds"]  # trials came from disk


def test_float_twin_is_shared_across_bit_widths_of_a_tier(completed):
    from src.benchmark import phase9_grid

    config, config_path, _, _ = completed
    other = phase9_grid.run_configuration(config, config_path, "mlp_toy_bits3")
    assert other["training"]["float_twin_checkpoint_reused"] is True and other["training"]["qat_checkpoint_reused"] is False


def test_changing_a_hyperparameter_invalidates_the_qat_checkpoint(grid):
    from src.benchmark import phase9_grid

    config, _, _ = grid
    data = phase9_grid.load_tier_data(config, TIER)
    assert phase9_grid.train_qat_checkpointed(config, data, TIER, 4, 42)["reused"] is False
    assert phase9_grid.train_qat_checkpointed(config, data, TIER, 4, 42)["reused"] is True
    changed = {**config, "mlp": {**config["mlp"], "lr": 0.02}}
    assert phase9_grid.train_qat_checkpointed(changed, data, TIER, 4, 42)["reused"] is False


def test_key_memory_infeasibility_is_recorded_with_its_plaintext_gates(grid):
    from src.benchmark import phase9_grid

    config, config_path, _ = grid
    result = phase9_grid.run_configuration({**config, "max_key_material_gb": 1e-9}, config_path, "mlp_toy_bits4")
    assert result["status"] == "infeasible_key_memory" and "exceeds" in result["infeasible_reason"]
    assert result["gates_passed"]["t0"] is True and result["gates_passed"]["t2"] is None and result["all_gates_passed"] is False
    assert "fhe_latency" not in result and result["key_material_gb"] > 0 and result["circuit_stats"]["programmable_bootstrap_count"] > 0


def test_a_missing_tier_is_recorded_as_a_failure_not_raised(grid):
    from src.benchmark import phase9_grid

    config, config_path, out_dir = grid
    config = {**config, "configurations": [{"label": "mlp_missing_bits4", "tier": "missing", "n_bits": 4}]}
    result = phase9_grid.run_configuration(config, config_path, "mlp_missing_bits4")
    assert result["status"] == "train_failed" and result["error"]["stage"] == "train" and "Traceback" in result["error"]["traceback"]
    assert json.loads((out_dir / "mlp_missing_bits4" / "metrics.json").read_text())["status"] == "train_failed"


def test_plaintext_evaluation_reports_qat_and_float_twin_without_fhe(grid):
    from src.benchmark import phase9_grid

    config, _, _ = grid
    report = phase9_grid.plaintext_evaluation(config, TIER, 4, 42)
    for key in ("qat_clear_quantized", "float_twin"):
        assert 0.0 <= report[key]["pr_auc"] <= 1.0 and report[key]["confusion_matrix"] and report[key]["epochs_run"] >= 1
    assert report["test_partition_touched"] is False


def test_plaintext_seed_stability_summarizes_the_spread(grid, monkeypatch):
    from src.benchmark import phase9_grid

    config, _, _ = grid
    monkeypatch.setattr(phase9_grid, "run_plaintext", phase9_grid.run_plaintext)
    runs = [phase9_grid.plaintext_evaluation(config, TIER, 4, s) for s in config["seed_stability"]["seeds"]]
    assert [r["seed"] for r in runs] == [42, 43]


def test_write_summary_lists_every_configuration_including_not_run(completed):
    from src.benchmark import phase9_grid

    config, _, out_dir, _ = completed
    summary = phase9_grid.write_summary(config)
    by_label = {r["label"]: r for r in summary["configurations"]}
    assert by_label["mlp_toy_bits3"]["status"] == "not_run" and by_label["mlp_toy_bits4"]["status"] in ("passed", "failed_accuracy_gates")
    assert summary["all_run"] is False and (out_dir / "summary.json").exists()
