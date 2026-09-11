"""Phase 4 pipeline wiring test on synthetic data.

Exercises src.features.pipeline.run() end-to-end against a small synthetic
Phase2Features object PLUS a genuinely fitted, internally-consistent fake
"Phase 3" (model artifacts + metrics.json + provenance.json) -- proves the
integrity gate, ranking, tiering, and per-tier evaluation are wired
together correctly without needing the real dataset or a real Phase 3 run.
tests/test_features_pipeline_real_data.py is the real-data counterpart.

test_test_partition_never_touched mirrors
tests/test_train_pipeline_integration.py's mechanical proof: X_test/y_test
are sentinels that raise on any access besides len(), and the pipeline
must still complete successfully.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import provenance as data_provenance
from src.data.split import compute_split_boundaries
from src.features.pipeline import Phase3IntegrityError, run
from src.train import metrics as train_metrics
from src.train.data import Phase2Features
from src.train.imbalance import compute_scale_pos_weight
from src.train.logistic_regression import fit_logistic_regression
from src.train.xgboost_model import build_xgb_classifier

N_FEATURES = 20


def _make_synthetic_features(n=1200, seed=0) -> Phase2Features:
    rng = np.random.RandomState(seed)
    dt = pd.Series(np.arange(n) * 100)
    signal = rng.rand(n)
    y = pd.Series((signal > 0.9).astype(int))
    data = {"f0": signal.astype("float32")}
    for i in range(1, N_FEATURES):
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
    train_dt = dt.loc[train_mask]
    X_train_plus_val = X.loc[train_plus_val_mask]
    y_train_plus_val = y.loc[train_plus_val_mask]
    train_plus_val_dt = dt.loc[train_plus_val_mask]

    return Phase2Features(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X.iloc[:5],
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
    project allows (see src/features/pipeline.py's split_sizes)."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self):
        return self._length

    def __getattr__(self, name):
        raise AssertionError(f"Test partition was accessed via .{name} -- it must stay untouched")

    def __getitem__(self, item):
        raise AssertionError("Test partition was indexed -- it must stay untouched")


def _write_fake_phase3(tmp_path: Path, features: Phase2Features, phase3_config: dict) -> None:
    """Genuinely fits+saves a Phase 3 baseline the way src/train/pipeline.py
    would, so Phase 4's integrity gate sees an internally consistent
    committed state -- not a hand-typed placeholder metrics.json that
    would pass the gate for the wrong reason."""
    seed = phase3_config["seed"]
    c = phase3_config["models"]["logistic_regression"]["c_grid"][0]
    max_iter = phase3_config["models"]["logistic_regression"]["max_iter"]
    max_depth = phase3_config["models"]["xgboost"]["max_depth_grid"][0]
    learning_rate = phase3_config["models"]["xgboost"]["learning_rate_grid"][0]

    lr_model = fit_logistic_regression(features.X_train.to_numpy(), features.y_train.to_numpy(), c, seed, max_iter)
    xgb_model = build_xgb_classifier(
        max_depth, learning_rate, compute_scale_pos_weight(features.y_train.to_numpy()), seed
    )
    xgb_model.fit(
        features.X_train.to_numpy(),
        features.y_train.to_numpy(),
        eval_set=[(features.X_val.to_numpy(), features.y_val.to_numpy())],
        verbose=False,
    )

    lr_prob = lr_model.predict_proba(features.X_val.to_numpy())[:, 1]
    xgb_prob = xgb_model.predict_proba(features.X_val.to_numpy())[:, 1]

    output_dir = tmp_path / phase3_config["output"]["dir"]
    models_dir = tmp_path / phase3_config["output"]["models_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    xgb_model.save_model(str(models_dir / "xgboost.json"))

    metrics = {
        "logistic_regression": {
            "selected_C": c,
            "val_evaluation": train_metrics.full_evaluation(features.y_val.to_numpy(), lr_prob),
        },
        "xgboost": {
            "selected_max_depth": max_depth,
            "selected_learning_rate": learning_rate,
            "val_evaluation": train_metrics.full_evaluation(features.y_val.to_numpy(), xgb_prob),
        },
        "test_partition_touched": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, default=str))

    provenance = data_provenance.build_provenance(
        config=phase3_config,
        config_path=Path("configs/phase3/baselines.yaml"),
        seed=seed,
        raw_file_digests={},
        library_names=["numpy"],
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, default=str))


@pytest.fixture
def phase4_setup(tmp_path: Path, monkeypatch):
    features = _make_synthetic_features()

    phase3_config = {
        "phase": 3,
        "seed": 42,
        "phase2_config": "configs/phase2/pipeline.yaml",
        "models": {
            "logistic_regression": {"c_grid": [1.0], "max_iter": 200},
            "xgboost": {"max_depth_grid": [3], "learning_rate_grid": [0.1]},
        },
        "error_analysis": {"threshold_metric": "f1"},
        "output": {"dir": "results/phase3_baselines", "models_dir": "models/phase3_baselines"},
    }
    phase3_config_dir = tmp_path / "configs" / "phase3"
    phase3_config_dir.mkdir(parents=True, exist_ok=True)
    phase3_config_path = phase3_config_dir / "baselines.yaml"
    phase3_config_path.write_text(yaml.safe_dump(phase3_config))

    _write_fake_phase3(tmp_path, features, phase3_config)

    blocks_dir = tmp_path / "results" / "phase1_eda"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    (blocks_dir / "v_null_mask_blocks.csv").write_text(
        'block_id,null_mask_sha256,n_columns,columns\n0,abc,3,"f5,f6,f7"\n'
    )

    phase4_config_dict = {
        "phase": 4,
        "seed": 42,
        "phase2_config": "configs/phase2/pipeline.yaml",
        "phase3_config": "configs/phase3/baselines.yaml",
        "importance": {"method": "xgboost_total_gain", "ranking_stability_seeds": [43, 44]},
        "tiers": {"sizes": [2, 5, 8]},
        "curve": {"feature_counts": [2, 5, 8]},
        "stability": {"tier_eval_seeds": [42, 43]},
        # Tight: the reproduction check refits in Phase 2's OWN column order
        # (feature order in `features`), matching _write_fake_phase3's fit --
        # see configs/phase4/features.yaml's reproduction_check comment for
        # the controlled experiment showing this reproduces to ~0.0 diff,
        # unlike a rank-order refit (an earlier, incorrect version of this
        # check misattributed that order-driven diff to thread nondeterminism).
        "reproduction_check": {"pr_auc_tolerance": 1.0e-6},
        "output": {"dir": "results/phase4_features", "models_dir": "models/phase4_features"},
    }
    phase4_config_dir = tmp_path / "configs" / "phase4"
    phase4_config_dir.mkdir(parents=True, exist_ok=True)
    phase4_config_path = phase4_config_dir / "features.yaml"
    phase4_config_path.write_text(yaml.safe_dump(phase4_config_dict))

    import src.config as config_module
    import src.features.pipeline as pipeline_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_module, "load_phase2_features", lambda cfg_path: features)
    # src/features/pipeline.py::run() calls load_config(config["phase3_config"])
    # directly (unlike load_phase2_features, which is monkeypatched away
    # above) -- and that path is written "configs/phase3/baselines.yaml"
    # per this project's established config-cross-reference convention
    # (e.g. configs/phase3/baselines.yaml's own "phase2_config" field).
    # src.config.load_config's CONFIGS_DIR-relative lookup double-prefixes
    # such a path (CONFIGS_DIR already ends in "configs"), so it silently
    # falls through to a CWD-relative fallback -- which is also exactly
    # what happens in real usage (scripts are run via `python -m
    # src.train.pipeline` from the project root, i.e. CWD == PROJECT_ROOT).
    # chdir into tmp_path here reproduces that same real assumption instead
    # of relying on the CONFIGS_DIR patch alone, which is NOT sufficient
    # for this particular call site.
    monkeypatch.chdir(tmp_path)

    return str(phase4_config_path), tmp_path, features


def test_pipeline_runs_end_to_end_on_synthetic_data(phase4_setup):
    config_path, _, _ = phase4_setup
    result = run(config_path)
    assert result["test_partition_touched"] is False
    assert set(result["tiers"].keys()) == {"top_2", "top_5", "top_8"}


def test_pipeline_writes_expected_output_files(phase4_setup):
    config_path, tmp_path, _ = phase4_setup
    run(config_path)
    output_dir = tmp_path / "results" / "phase4_features"
    for name in (
        "feature_ranking.csv",
        "ranking_stability.csv",
        "accuracy_vs_feature_count.csv",
        "tiers.json",
        "metrics.json",
        "provenance.json",
    ):
        assert (output_dir / name).exists(), name


def test_pipeline_saves_per_tier_model_artifacts(phase4_setup):
    config_path, tmp_path, _ = phase4_setup
    run(config_path)
    models_dir = tmp_path / "models" / "phase4_features"
    for tier in ("top_2", "top_5", "top_8"):
        assert (models_dir / tier / "logistic_regression.joblib").exists()
        assert (models_dir / tier / "xgboost.json").exists()


def test_pipeline_tiers_are_nested_and_hash_verified(phase4_setup):
    config_path, tmp_path, _ = phase4_setup
    run(config_path)
    from src.features.tiers import load_tiers

    payload = load_tiers(tmp_path / "results" / "phase4_features" / "tiers.json")
    tiers = payload["tiers"]
    assert tiers["top_2"] == tiers["top_5"][:2]
    assert tiers["top_5"] == tiers["top_8"][:5]


def test_pipeline_fails_loudly_when_phase3_config_changed_since_commit(phase4_setup):
    config_path, tmp_path, _ = phase4_setup
    phase3_path = tmp_path / "configs" / "phase3" / "baselines.yaml"
    cfg = yaml.safe_load(phase3_path.read_text())
    cfg["models"]["logistic_regression"]["c_grid"] = [999.0]
    phase3_path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(Phase3IntegrityError):
        run(config_path)


def test_pipeline_fails_loudly_when_phase3_model_missing(phase4_setup):
    config_path, tmp_path, _ = phase4_setup
    (tmp_path / "models" / "phase3_baselines" / "xgboost.json").unlink()

    with pytest.raises(Phase3IntegrityError):
        run(config_path)


def test_test_partition_never_touched(phase4_setup, monkeypatch):
    config_path, tmp_path, features = phase4_setup
    import src.features.pipeline as pipeline_module

    exploding_features = Phase2Features(
        X_train=features.X_train,
        y_train=features.y_train,
        X_val=features.X_val,
        y_val=features.y_val,
        X_test=_ExplodingSentinel(37),
        y_test=_ExplodingSentinel(37),
        train_dt=features.train_dt,
        boundaries=features.boundaries,
        X_train_plus_val=features.X_train_plus_val,
        y_train_plus_val=features.y_train_plus_val,
        train_plus_val_dt=features.train_plus_val_dt,
    )
    monkeypatch.setattr(pipeline_module, "load_phase2_features", lambda cfg_path: exploding_features)

    result = run(config_path)  # must not raise
    assert result["split_sizes"]["test"] == 37
