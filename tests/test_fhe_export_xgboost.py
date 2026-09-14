"""Phase 6 XGBoost export wiring test on synthetic data (Windows-side).

Exercises src.fhe.export_xgboost.run() end-to-end against a small synthetic
Phase2Features object plus a genuinely fitted, early-stopped fake "committed
Phase 4" (per-tier xgboost.json + metrics.json + provenance.json + tiers.json).
The fake boosters are fit so they store more trees than best_iteration+1,
which is what makes the slicing assertions meaningful.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml

from src.data import provenance as data_provenance
from src.data.split import compute_split_boundaries
from src.features.tiers import save_tiers
from src.fhe.export_xgboost import Phase4XGBIntegrityError, run
from src.fhe.handoff import load_handoff, sha256_file
from src.train import metrics as train_metrics
from src.train.data import Phase2Features
from src.train.imbalance import compute_scale_pos_weight
from src.train.xgboost_model import build_xgb_classifier

TIERS = {"top_2": ["f2", "f0"], "top_3": ["f2", "f0", "f4"]}  # deliberately NOT Phase 2 column order


def _make_synthetic_features(n=1500, seed=0) -> Phase2Features:
    rng = np.random.RandomState(seed)
    dt = pd.Series(np.arange(n) * 100)
    f0 = rng.rand(n)
    f2 = rng.rand(n)
    y = pd.Series(((f0 + 0.5 * f2 + 0.3 * rng.rand(n)) > 1.2).astype(int))
    data = {f"f{i}": rng.rand(n).astype("float32") for i in range(5)}
    data["f0"] = (f0 * 1000.0).astype("float32")
    data["f2"] = f2.astype("float32")
    X = pd.DataFrame(data, index=pd.Index(range(1, n + 1), name="TransactionID"))
    y.index = X.index
    dt.index = X.index

    boundaries = compute_split_boundaries(
        dt, {"train_val_quantile": 0.6, "val_test_quantile": 0.97, "cv_n_folds": 2}
    )
    train_mask = (dt <= boundaries.train_val_boundary).to_numpy()
    val_mask = ((dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)).to_numpy()
    train_plus_val_mask = train_mask | val_mask

    return Phase2Features(
        X_train=X.loc[train_mask], y_train=y.loc[train_mask], X_val=X.loc[val_mask], y_val=y.loc[val_mask],
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
    output_dir = tmp_path / phase4_config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    y_train, y_val = features.y_train.to_numpy(), features.y_val.to_numpy()

    tier_metrics = {}
    for tier, columns in TIERS.items():
        model = build_xgb_classifier(
            max_depth=2, learning_rate=0.3, scale_pos_weight=compute_scale_pos_weight(y_train), seed=42, n_estimators=300
        )
        X_val = features.X_val[columns].to_numpy()
        model.fit(features.X_train[columns].to_numpy(), y_train, eval_set=[(X_val, y_val)], verbose=False)
        n_stored = model.get_booster().num_boosted_rounds()
        assert n_stored > model.best_iteration + 1, "fixture must store trees beyond best_iteration"

        tier_dir = tmp_path / phase4_config["output"]["models_dir"] / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(str(tier_dir / "xgboost.json"))

        tier_metrics[tier] = {
            "xgboost": {
                "selected_max_depth": 2,
                "selected_learning_rate": 0.3,
                "selected_n_estimators": int(model.best_iteration) + 1,
                "val_evaluation": train_metrics.full_evaluation(y_val, model.predict_proba(X_val)[:, 1]),
            }
        }

    (output_dir / "metrics.json").write_text(json.dumps({"tiers": tier_metrics}, default=str))
    provenance = data_provenance.build_provenance(
        config=phase4_config, config_path=Path("configs/phase4/features.yaml"),
        seed=phase4_config["seed"], raw_file_digests={}, library_names=["numpy"],
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, default=str))
    save_tiers(output_dir / "tiers.json", TIERS, {"importance_method": "xgboost_total_gain", "seed": 42})


@pytest.fixture
def phase6_setup(tmp_path: Path, monkeypatch):
    features = _make_synthetic_features()
    phase4_config = {
        "phase": 4, "seed": 42,
        "output": {"dir": "results/phase4_features", "models_dir": "models/phase4_features"},
    }
    (tmp_path / "configs" / "phase4").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "phase4" / "features.yaml").write_text(yaml.safe_dump(phase4_config))
    _write_fake_phase4(tmp_path, features, phase4_config)

    phase6_config = {
        "phase": 6, "seed": 42,
        "phase2_config": "configs/phase2/pipeline.yaml",
        "phase4_config": "configs/phase4/features.yaml",
        "tiers": list(TIERS),
        "n_bits": 8,
        "phase4_integrity_tolerance": 1.0e-6,
        "output": {"dir": "results/phase6_fhe_xgboost", "handoff_dir": "data/fhe_handoff/phase6"},
    }
    (tmp_path / "configs" / "phase6").mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "configs" / "phase6" / "xgb_poc.yaml"
    config_path.write_text(yaml.safe_dump(phase6_config))

    import src.config as config_module
    import src.fhe.export_xgboost as export_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(export_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "load_phase2_features", lambda cfg_path: features)
    monkeypatch.chdir(tmp_path)  # same load_config resolution note as tests/test_fhe_export.py
    return str(config_path), tmp_path, features


def _load_tier(tmp_path: Path, tier: str):
    return load_handoff(
        tmp_path / "data" / "fhe_handoff" / "phase6" / f"xgb_{tier}.npz",
        tmp_path / "results" / "phase6_fhe_xgboost" / tier / "handoff_manifest.json",
    )


def test_export_runs_every_tier(phase6_setup):
    config_path, _, _ = phase6_setup
    summary = run(config_path)
    assert [t["tier"] for t in summary["tiers"]] == list(TIERS)
    assert [t["n_features"] for t in summary["tiers"]] == [len(c) for c in TIERS.values()]


def test_export_writes_handoff_in_tier_column_order(phase6_setup):
    config_path, tmp_path, features = phase6_setup
    run(config_path)
    for tier, columns in TIERS.items():
        manifest, arrays = _load_tier(tmp_path, tier)
        assert manifest["columns"] == columns
        np.testing.assert_array_equal(arrays["X_val"], features.X_val[columns].to_numpy())
        np.testing.assert_array_equal(arrays["X_train"], features.X_train[columns].to_numpy())


def test_exported_booster_is_sliced_to_inference_trees_and_reproduces_reference(phase6_setup):
    config_path, tmp_path, _ = phase6_setup
    run(config_path)
    for tier in TIERS:
        manifest, arrays = _load_tier(tmp_path, tier)
        booster = xgb.Booster()
        booster.load_model(tmp_path / manifest["booster_path"])
        assert booster.num_boosted_rounds() == manifest["n_trees_inference"] == manifest["best_iteration"] + 1
        assert manifest["n_trees_stored_in_phase4_booster"] > manifest["n_trees_inference"]
        np.testing.assert_array_equal(booster.inplace_predict(arrays["X_val"]), arrays["reference_val_prob"])
        assert sha256_file(tmp_path / manifest["booster_path"]) == manifest["booster_sha256"]


def test_reference_matches_unsliced_phase4_model_default_prediction(phase6_setup):
    """The reference must be exactly what the committed Phase 4 model predicts by
    default (best_iteration applied), not a prediction over all stored trees."""
    config_path, tmp_path, features = phase6_setup
    run(config_path)
    for tier, columns in TIERS.items():
        _, arrays = _load_tier(tmp_path, tier)
        model = xgb.XGBClassifier()
        model.load_model(tmp_path / "models" / "phase4_features" / tier / "xgboost.json")
        np.testing.assert_array_equal(
            model.predict_proba(features.X_val[columns].to_numpy())[:, 1], arrays["reference_val_prob"]
        )


def test_export_fails_loudly_when_phase4_config_changed_since_commit(phase6_setup):
    config_path, tmp_path, _ = phase6_setup
    path = tmp_path / "configs" / "phase4" / "features.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["seed"] = 999
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(Phase4XGBIntegrityError, match="config_hash mismatch"):
        run(config_path)


def test_export_fails_loudly_when_tier_model_missing(phase6_setup):
    config_path, tmp_path, _ = phase6_setup
    (tmp_path / "models" / "phase4_features" / "top_3" / "xgboost.json").unlink()
    with pytest.raises(Phase4XGBIntegrityError, match="tier model missing"):
        run(config_path)


def test_export_fails_loudly_when_committed_n_estimators_disagrees(phase6_setup):
    config_path, tmp_path, _ = phase6_setup
    metrics_path = tmp_path / "results" / "phase4_features" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["tiers"]["top_2"]["xgboost"]["selected_n_estimators"] += 1
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(Phase4XGBIntegrityError, match="selected_n_estimators"):
        run(config_path)


def test_test_partition_never_touched(phase6_setup, monkeypatch):
    config_path, _, features = phase6_setup
    import src.fhe.export_xgboost as export_module

    exploding = Phase2Features(
        X_train=features.X_train, y_train=features.y_train, X_val=features.X_val, y_val=features.y_val,
        X_test=_ExplodingSentinel(37), y_test=_ExplodingSentinel(37),
        train_dt=features.train_dt, boundaries=features.boundaries,
        X_train_plus_val=features.X_train_plus_val, y_train_plus_val=features.y_train_plus_val,
        train_plus_val_dt=features.train_plus_val_dt,
    )
    monkeypatch.setattr(export_module, "load_phase2_features", lambda cfg_path: exploding)
    summary = run(config_path)
    assert summary["n_test"] == 37
    assert summary["test_partition_touched"] is False
