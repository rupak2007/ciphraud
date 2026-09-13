"""Phase 5 export wiring test on synthetic data (Windows-side).

Exercises src.fhe.export.run() end-to-end against a small synthetic
Phase2Features object PLUS a genuinely fitted, internally-consistent fake
"committed Phase 4" (tier model + metrics.json + provenance.json +
tiers.json) -- proves the integrity gate and the handoff export are wired
together correctly without needing the real dataset or a real Phase 4 run.

test_test_partition_never_touched mirrors the same mechanical proof used
in tests/test_train_pipeline_integration.py and
tests/test_features_pipeline_integration.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import provenance as data_provenance
from src.data.split import compute_split_boundaries
from src.fhe.export import Phase4IntegrityError, run
from src.fhe.handoff import load_handoff, rebuild_pipeline_predict_proba
from src.features.tiers import save_tiers
from src.train import metrics as train_metrics
from src.train.data import Phase2Features

TIER_COLUMNS = ["f2", "f0", "f4"]  # deliberately NOT Phase2 column order
ALL_COLUMNS = ["f0", "f1", "f2", "f3", "f4"]


def _make_synthetic_features(n=800, seed=0) -> Phase2Features:
    rng = np.random.RandomState(seed)
    dt = pd.Series(np.arange(n) * 100)
    signal = rng.rand(n)
    y = pd.Series((signal > 0.9).astype(int))
    data = {"f0": signal.astype("float32") * 1000.0}  # deliberately unscaled
    for i in range(1, 5):
        data[f"f{i}"] = rng.rand(n).astype("float32")
    X = pd.DataFrame(data, index=pd.Index(range(1, n + 1), name="TransactionID"))
    y.index = X.index
    dt.index = X.index

    boundaries = compute_split_boundaries(
        dt, {"train_val_quantile": 0.6, "val_test_quantile": 0.97, "cv_n_folds": 2}
    )
    train_mask = (dt <= boundaries.train_val_boundary).to_numpy()
    val_mask = ((dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)).to_numpy()
    train_plus_val_mask = train_mask | val_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]

    return Phase2Features(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        X_test=X.iloc[:5], y_test=y.iloc[:5],
        train_dt=dt.loc[train_mask], boundaries=boundaries,
        X_train_plus_val=X.loc[train_plus_val_mask], y_train_plus_val=y.loc[train_plus_val_mask],
        train_plus_val_dt=dt.loc[train_plus_val_mask],
    )


class _ExplodingSentinel:
    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length

    def __getattr__(self, name):
        raise AssertionError(f"Test partition was accessed via .{name} -- it must stay untouched")

    def __getitem__(self, item):
        raise AssertionError("Test partition was indexed -- it must stay untouched")


def _write_fake_phase4(tmp_path: Path, features: Phase2Features, phase4_config: dict) -> None:
    """Genuinely fits+saves a Phase 4 top_20-shaped tier model in
    TIER_COLUMNS order (not ALL_COLUMNS order), so export.py's
    column-order handling is exercised, not just its happy path."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(features.X_train[TIER_COLUMNS].to_numpy(), features.y_train.to_numpy())
    prob = pipeline.predict_proba(features.X_val[TIER_COLUMNS].to_numpy())[:, 1]

    output_dir = tmp_path / phase4_config["output"]["dir"]
    models_dir = tmp_path / phase4_config["output"]["models_dir"] / "top_20"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / "logistic_regression.joblib")

    metrics = {
        "tiers": {
            "top_20": {
                "logistic_regression": {
                    "selected_C": 1.0,
                    "val_evaluation": train_metrics.full_evaluation(features.y_val.to_numpy(), prob),
                }
            }
        },
        "test_partition_touched": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, default=str))

    provenance = data_provenance.build_provenance(
        config=phase4_config, config_path=Path("configs/phase4/features.yaml"),
        seed=phase4_config["seed"], raw_file_digests={}, library_names=["numpy"],
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, default=str))

    save_tiers(
        output_dir / "tiers.json",
        {"top_20": TIER_COLUMNS},
        {"importance_method": "xgboost_total_gain", "seed": phase4_config["seed"]},
    )


@pytest.fixture
def phase5_setup(tmp_path: Path, monkeypatch):
    features = _make_synthetic_features()
    phase4_config = {
        "phase": 4, "seed": 42,
        "output": {"dir": "results/phase4_features", "models_dir": "models/phase4_features"},
    }
    phase4_config_dir = tmp_path / "configs" / "phase4"
    phase4_config_dir.mkdir(parents=True, exist_ok=True)
    (phase4_config_dir / "features.yaml").write_text(yaml.safe_dump(phase4_config))
    _write_fake_phase4(tmp_path, features, phase4_config)

    phase5_config = {
        "phase": 5, "seed": 42,
        "phase2_config": "configs/phase2/pipeline.yaml",
        "phase4_config": "configs/phase4/features.yaml",
        "tier": "top_20",
        "n_bits": 8,
        "phase4_integrity_tolerance": 1.0e-6,
        "execute_sample": {"n": 10, "n_positive": 3, "split": "val"},
        "tolerances": {
            "t0_max_abs_diff": 1.0e-9, "t3_min_decision_agreement": 0.99, "t3_max_pr_auc_drop": 0.01,
        },
        "output": {
            "handoff_npz": "data/fhe_handoff/lr_top20.npz",
            "handoff_manifest": "results/phase5_fhe_poc/handoff_manifest.json",
            "params_path": "results/phase5_fhe_poc/lr_top20_params.json",
            "dir": "results/phase5_fhe_poc",
        },
    }
    phase5_config_dir = tmp_path / "configs" / "phase5"
    phase5_config_dir.mkdir(parents=True, exist_ok=True)
    phase5_config_path = phase5_config_dir / "lr_poc.yaml"
    phase5_config_path.write_text(yaml.safe_dump(phase5_config))

    import src.config as config_module
    import src.fhe.export as export_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(export_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "load_phase2_features", lambda cfg_path: features)
    # See docs/features.md Sec.6.2 -- load_config's CONFIGS_DIR-relative
    # resolution double-prefixes a conventionally-written
    # "configs/phase4/features.yaml" path, falling through to a
    # CWD-relative lookup that matches real usage (run via `python -m`
    # from the project root). chdir reproduces that here.
    monkeypatch.chdir(tmp_path)

    return str(phase5_config_path), tmp_path, features


def test_export_runs_end_to_end_on_synthetic_data(phase5_setup):
    config_path, _, _ = phase5_setup
    summary = run(config_path)
    assert summary["tier"] == "top_20"
    assert summary["n_features"] == 3


def test_export_writes_handoff_in_tier_column_order(phase5_setup):
    config_path, tmp_path, features = phase5_setup
    run(config_path)

    manifest, arrays = load_handoff(
        tmp_path / "data" / "fhe_handoff" / "lr_top20.npz", tmp_path / "results" / "phase5_fhe_poc" / "handoff_manifest.json"
    )
    assert manifest["columns"] == TIER_COLUMNS
    np.testing.assert_array_equal(arrays["X_val"], features.X_val[TIER_COLUMNS].to_numpy())
    np.testing.assert_array_equal(arrays["X_train"], features.X_train[TIER_COLUMNS].to_numpy())


def test_export_params_reproduce_reference_probabilities(phase5_setup):
    """The whole point of the handoff: params + rebuild_pipeline_predict_proba
    must reproduce the SAME reference probabilities the export wrote,
    exactly -- this is T0's actual mechanism, exercised here on synthetic
    data before any WSL/Concrete-ML involvement."""
    config_path, tmp_path, _ = phase5_setup
    run(config_path)

    params = json.loads((tmp_path / "results" / "phase5_fhe_poc" / "lr_top20_params.json").read_text())
    manifest, arrays = load_handoff(
        tmp_path / "data" / "fhe_handoff" / "lr_top20.npz", tmp_path / "results" / "phase5_fhe_poc" / "handoff_manifest.json"
    )
    rebuilt = rebuild_pipeline_predict_proba(params, arrays["X_val"])[:, 1]
    np.testing.assert_allclose(rebuilt, arrays["reference_val_prob"], atol=1e-9)


def test_export_fails_loudly_when_phase4_config_changed_since_commit(phase5_setup):
    config_path, tmp_path, _ = phase5_setup
    phase4_config_path = tmp_path / "configs" / "phase4" / "features.yaml"
    cfg = yaml.safe_load(phase4_config_path.read_text())
    cfg["seed"] = 999  # mutate after the committed provenance was written
    phase4_config_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(Phase4IntegrityError, match="config_hash mismatch"):
        run(config_path)


def test_export_fails_loudly_when_phase4_model_missing(phase5_setup):
    config_path, tmp_path, _ = phase5_setup
    (tmp_path / "models" / "phase4_features" / "top_20" / "logistic_regression.joblib").unlink()

    with pytest.raises(Phase4IntegrityError, match="artifacts missing"):
        run(config_path)


def test_test_partition_never_touched(phase5_setup, monkeypatch):
    config_path, _, features = phase5_setup
    import src.fhe.export as export_module

    exploding_features = Phase2Features(
        X_train=features.X_train, y_train=features.y_train,
        X_val=features.X_val, y_val=features.y_val,
        X_test=_ExplodingSentinel(37), y_test=_ExplodingSentinel(37),
        train_dt=features.train_dt, boundaries=features.boundaries,
        X_train_plus_val=features.X_train_plus_val, y_train_plus_val=features.y_train_plus_val,
        train_plus_val_dt=features.train_plus_val_dt,
    )
    monkeypatch.setattr(export_module, "load_phase2_features", lambda cfg_path: exploding_features)

    summary = run(config_path)  # must not raise
    assert summary["n_test"] == 37  # len() was the only permitted access
    assert summary["test_partition_touched"] is False
