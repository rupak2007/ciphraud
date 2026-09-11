"""Phase 3 pipeline wiring test on synthetic data.

Exercises src.train.pipeline.run() end-to-end against a small synthetic
Phase2Features object (monkeypatching load_phase2_features) -- proves the
modules (data, logistic_regression, xgboost_model, metrics) are wired
together correctly without needing the real dataset or a real Phase 2
run. tests/test_train_pipeline_real_data.py is the real-data counterpart.

test_test_partition_never_touched is the most important test in this
file: it passes a sentinel object as X_test/y_test that raises on any
access except __len__, and asserts src.train.pipeline.run() completes
successfully without ever touching it -- a direct, mechanical proof of
the "test set stays untouched" requirement, not just a code-review claim.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.split import SplitBoundaries, compute_split_boundaries
from src.train.data import Phase2Features
from src.train.pipeline import run


def _make_synthetic_features(n=600, n_val=150, seed=0) -> Phase2Features:
    rng = np.random.RandomState(seed)
    n_total = n + n_val
    dt = pd.Series(np.arange(n_total) * 100)
    signal = rng.rand(n_total)
    y = pd.Series((signal > 0.9).astype(int))
    X = pd.DataFrame(
        {
            "signal_feature": signal.astype("float32"),
            "noise_feature": rng.rand(n_total).astype("float32"),
        },
        index=pd.Index(range(1, n_total + 1), name="TransactionID"),
    )
    y.index = X.index
    dt.index = X.index

    # val_test_quantile < 1.0 leaves a small genuine "test" sliver, so
    # train/val/test/cv all have distinct, correctly-bounded ranges here --
    # the same shape as the real Phase2Features, which is what let the
    # real train+val-vs-train-only bug (docs/baselines.md Sec.6/Sec.8)
    # hide from a less careful synthetic fixture.
    boundaries = compute_split_boundaries(
        dt, {"train_val_quantile": n / n_total, "val_test_quantile": 0.97, "cv_n_folds": 2}
    )
    train_mask = (dt <= boundaries.train_val_boundary).to_numpy()
    val_mask = ((dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)).to_numpy()
    train_plus_val_mask = train_mask | val_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    train_dt = dt.loc[train_mask]
    X_train_plus_val = X.loc[train_plus_val_mask]
    y_train_plus_val = y.loc[train_plus_val_mask]
    train_plus_val_dt = dt.loc[train_plus_val_mask]

    return Phase2Features(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X.iloc[:5],  # placeholder; replaced with a sentinel in the key test
        y_test=y.iloc[:5],
        train_dt=train_dt,
        boundaries=boundaries,
        X_train_plus_val=X_train_plus_val,
        y_train_plus_val=y_train_plus_val,
        train_plus_val_dt=train_plus_val_dt,
    )


class _ExplodingSentinel:
    """Raises on virtually any use except len() -- reporting the test
    partition's row count is the one legitimate, content-free use this
    project allows (see src/train/pipeline.py's split_sizes)."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length

    def __getattr__(self, name):
        raise AssertionError(f"Test partition was accessed via .{name} -- it must stay untouched")

    def __getitem__(self, item):
        raise AssertionError("Test partition was indexed -- it must stay untouched")


@pytest.fixture
def synthetic_config(tmp_path: Path, monkeypatch) -> tuple[str, Phase2Features]:
    features = _make_synthetic_features()

    config = {
        "phase": 3,
        "seed": 42,
        "phase2_config": "configs/phase2/pipeline.yaml",
        "models": {
            "logistic_regression": {"c_grid": [0.1, 1.0]},
            "xgboost": {"max_depth_grid": [3], "learning_rate_grid": [0.1]},
        },
        "error_analysis": {"threshold_metric": "f1"},
        "output": {"dir": "results/phase3_baselines", "models_dir": "models/phase3_baselines"},
    }
    config_dir = tmp_path / "configs" / "phase3"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "baselines.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import src.config as config_module
    import src.train.pipeline as pipeline_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_module, "load_phase2_features", lambda cfg_path: features)

    return str(config_path), tmp_path


def test_pipeline_runs_end_to_end_on_synthetic_data(synthetic_config):
    config_path, _ = synthetic_config
    result = run(config_path)
    assert "logistic_regression" in result
    assert "xgboost" in result
    assert result["strongest_model"] in ("logistic_regression", "xgboost")


def test_pipeline_writes_expected_output_files(synthetic_config):
    config_path, tmp_path = synthetic_config
    run(config_path)
    output_dir = tmp_path / "results" / "phase3_baselines"
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "provenance.json").exists()


def test_pipeline_saves_both_model_artifacts(synthetic_config):
    config_path, tmp_path = synthetic_config
    run(config_path)
    models_dir = tmp_path / "models" / "phase3_baselines"
    assert (models_dir / "logistic_regression.joblib").exists()
    assert (models_dir / "xgboost.json").exists()


def test_pipeline_metrics_include_required_fields(synthetic_config):
    config_path, tmp_path = synthetic_config
    run(config_path)
    metrics = json.loads((tmp_path / "results" / "phase3_baselines" / "metrics.json").read_text())
    for model_key in ("logistic_regression", "xgboost"):
        val_eval = metrics[model_key]["val_evaluation"]
        assert "pr_auc" in val_eval
        assert "roc_auc" in val_eval
        assert "f1_selected" in val_eval
        assert "f2_selected" in val_eval
    assert metrics["test_partition_touched"] is False


def test_test_partition_never_touched(synthetic_config):
    """The mechanical proof: X_test/y_test are sentinels that raise on any
    access besides len(); the full pipeline must still complete."""
    config_path, tmp_path = synthetic_config

    import src.train.pipeline as pipeline_module

    base_features = _make_synthetic_features()
    exploding_features = Phase2Features(
        X_train=base_features.X_train,
        y_train=base_features.y_train,
        X_val=base_features.X_val,
        y_val=base_features.y_val,
        X_test=_ExplodingSentinel(37),
        y_test=_ExplodingSentinel(37),
        train_dt=base_features.train_dt,
        boundaries=base_features.boundaries,
        X_train_plus_val=base_features.X_train_plus_val,
        y_train_plus_val=base_features.y_train_plus_val,
        train_plus_val_dt=base_features.train_plus_val_dt,
    )
    pipeline_module.load_phase2_features = lambda cfg_path: exploding_features

    result = run(config_path)  # must not raise
    assert result["split_sizes"]["test"] == 37  # len() was the only permitted access
