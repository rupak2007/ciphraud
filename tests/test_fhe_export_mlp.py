"""Phase 9 MLP export tests (Windows-side): wiring on synthetic data, plus a check of the real exported handoffs.

Mirrors tests/test_fhe_export.py: a small synthetic `Phase2Features` and a hash-verified fake `tiers.json`.
`test_test_partition_never_touched` uses the same exploding-sentinel proof (stricter here: the MLP export does not
even call `len()` on the test partition).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.fhe.export_mlp import fit_train_only_scaler, run
from src.fhe.handoff import HandoffError, load_handoff, sha256_file, standardize_features
from src.features.tiers import save_tiers
from src.train.data import Phase2Features
from tests.test_fhe_export import TIER_COLUMNS, _ExplodingSentinel, _make_synthetic_features

REAL_HANDOFF = Path(__file__).resolve().parents[1] / "data" / "fhe_handoff" / "phase9"


@pytest.fixture
def mlp_setup(tmp_path: Path, monkeypatch):
    features = _make_synthetic_features()
    phase4_config = {"phase": 4, "seed": 42, "output": {"dir": "results/phase4_features", "models_dir": "models/phase4_features"}}
    (tmp_path / "configs" / "phase4").mkdir(parents=True)
    (tmp_path / "configs" / "phase4" / "features.yaml").write_text(yaml.safe_dump(phase4_config))
    (tmp_path / "results" / "phase4_features").mkdir(parents=True)
    save_tiers(
        tmp_path / "results" / "phase4_features" / "tiers.json",
        {"top_20": TIER_COLUMNS, "top_50": ["f1", "f3", "f0", "f2"]},
        {"importance_method": "xgboost_total_gain", "seed": 42},
    )
    mlp_config = {
        "phase": 9, "seed": 42, "phase2_config": "configs/phase2/pipeline.yaml", "phase4_config": "configs/phase4/features.yaml",
        "tiers": ["top_20", "top_50"], "output": {"handoff_dir": "data/fhe_handoff/phase9", "dir": "results/phase9_mlp/export"},
    }
    (tmp_path / "configs" / "phase9").mkdir(parents=True)
    config_path = tmp_path / "configs" / "phase9" / "mlp_export.yaml"
    config_path.write_text(yaml.safe_dump(mlp_config))

    import src.config as config_module
    import src.fhe.export_mlp as export_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(export_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export_module, "load_phase2_features", lambda cfg_path: features)
    monkeypatch.chdir(tmp_path)
    return str(config_path), tmp_path, features


def _load(tmp_path: Path, tier: str):
    return load_handoff(tmp_path / "data" / "fhe_handoff" / "phase9" / f"mlp_{tier}.npz", tmp_path / "results" / "phase9_mlp" / "export" / tier / "handoff_manifest.json")


def test_export_writes_every_tier_in_tier_column_order_with_labels(mlp_setup):
    config_path, tmp_path, features = mlp_setup
    summaries = run(config_path)
    assert [s["tier"] for s in summaries] == ["top_20", "top_50"]
    for tier, columns in (("top_20", TIER_COLUMNS), ("top_50", ["f1", "f3", "f0", "f2"])):
        manifest, arrays = _load(tmp_path, tier)
        assert manifest["columns"] == columns and manifest["tier"] == tier and manifest["test_partition_touched"] is False
        np.testing.assert_array_equal(arrays["X_train"], features.X_train[columns].to_numpy())
        np.testing.assert_array_equal(arrays["X_val"], features.X_val[columns].to_numpy())
        np.testing.assert_array_equal(arrays["y_train"], features.y_train.to_numpy())
        np.testing.assert_array_equal(arrays["y_val"], features.y_val.to_numpy())
        assert "reference_val_prob" not in arrays  # no Windows-side model exists for the MLP
        assert manifest["train_fraud_rate"] == pytest.approx(float(features.y_train.mean()))


def test_scaler_is_fit_on_train_only(mlp_setup):
    config_path, tmp_path, features = mlp_setup
    run(config_path)
    manifest, arrays = _load(tmp_path, "top_20")
    scaler_path = tmp_path / manifest["scaler_path"]
    assert sha256_file(scaler_path) == manifest["scaler_sha256"] and manifest["scaler_fit_on"] == "X_train only"
    params = json.loads(scaler_path.read_text())
    X_train = features.X_train[TIER_COLUMNS].to_numpy().astype(np.float64)
    np.testing.assert_allclose(params["scaler_mean"], X_train.mean(axis=0))
    np.testing.assert_allclose(params["scaler_scale"], X_train.std(axis=0))
    # a leak would show up as statistics that include validation rows
    X_all = np.vstack([X_train, features.X_val[TIER_COLUMNS].to_numpy().astype(np.float64)])
    assert not np.allclose(params["scaler_mean"], X_all.mean(axis=0))
    Z = standardize_features(params, arrays["X_train"])
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_scaler_handles_zero_variance_columns_like_sklearn():
    params = fit_train_only_scaler(np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]))
    assert params["scaler_scale"][1] == 1.0  # constant column: scale 1, never a division by zero
    assert np.isfinite(standardize_features(params, np.array([[2.0, 5.0]]))).all()


def test_export_rejects_nan_and_non_binary_labels(mlp_setup, monkeypatch):
    config_path, _, features = mlp_setup
    import src.fhe.export_mlp as export_module

    bad_x = features.X_train.copy()
    bad_x.iloc[0, 0] = np.nan
    monkeypatch.setattr(export_module, "load_phase2_features", lambda p: Phase2Features(**{**features.__dict__, "X_train": bad_x}))
    with pytest.raises(ValueError, match="NaN"):
        run(config_path)

    bad_y = features.y_train.copy()
    bad_y.iloc[0] = 2
    monkeypatch.setattr(export_module, "load_phase2_features", lambda p: Phase2Features(**{**features.__dict__, "y_train": bad_y}))
    with pytest.raises(ValueError, match="binary"):
        run(config_path)


def test_test_partition_never_touched(mlp_setup, monkeypatch):
    config_path, _, features = mlp_setup
    import src.fhe.export_mlp as export_module

    exploding = Phase2Features(**{**features.__dict__, "X_test": _ExplodingSentinel(37), "y_test": _ExplodingSentinel(37)})
    monkeypatch.setattr(export_module, "load_phase2_features", lambda p: exploding)
    summaries = run(config_path)  # must not raise
    assert all(s["test_partition_touched"] is False for s in summaries)


def test_tampered_handoff_is_rejected(mlp_setup):
    config_path, tmp_path, _ = mlp_setup
    run(config_path)
    npz = tmp_path / "data" / "fhe_handoff" / "phase9" / "mlp_top_20.npz"
    npz.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(HandoffError, match="SHA-256"):
        _load(tmp_path, "top_20")


# ---- the real exported handoffs (skipped when the git-ignored data/ directory has not been exported) -------------


@pytest.mark.skipif(not (REAL_HANDOFF / "mlp_top_20.npz").exists(), reason="Phase 9 MLP handoff not exported (data/ is git-ignored)")
@pytest.mark.parametrize("tier, lr_npz, lr_manifest", [
    ("top_20", "data/fhe_handoff/lr_top20.npz", "results/phase5_fhe_poc/handoff_manifest.json"),
    ("top_50", "data/fhe_handoff/phase8/lr_top50.npz", "results/phase8_research/lr_export/top_50/handoff_manifest.json"),
    ("top_100", "data/fhe_handoff/phase8/lr_top100.npz", "results/phase8_research/lr_export/top_100/handoff_manifest.json"),
])
def test_real_mlp_handoff_matches_the_phase8_lr_handoff_and_adds_labels(tier, lr_npz, lr_manifest):
    root = Path(__file__).resolve().parents[1]
    manifest, arrays = load_handoff(REAL_HANDOFF / f"mlp_{tier}.npz", root / "results" / "phase9_mlp" / "export" / tier / "handoff_manifest.json")
    lr_m, lr_arrays = load_handoff(root / lr_npz, root / lr_manifest)
    assert manifest["columns"] == lr_m["columns"]
    for key in ("X_train", "X_val", "y_val", "val_transaction_ids"):
        np.testing.assert_array_equal(arrays[key], lr_arrays[key])
    assert set(np.unique(arrays["y_train"])) == {0, 1} and arrays["y_train"].shape[0] == arrays["X_train"].shape[0]
    assert not np.isnan(arrays["X_train"]).any() and manifest["test_partition_touched"] is False
    assert 0.03 < float(arrays["y_train"].mean()) < 0.05  # the ~3.5% fraud rate, stated in CLAUDE.md
