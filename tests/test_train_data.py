"""Tests for src/train/data.py's cache-freshness logic.

Builds a fake Phase 2 output tree directly (not by running the real
pipeline) so these tests are fast and synthetic, while still exercising
the exact freshness-check code path load_phase2_features uses.
"""

import json

import pandas as pd
import pytest
import yaml

from src.data import provenance as data_provenance
from src.data.split import SplitBoundaries


def _write_fake_phase2_output(tmp_path, config: dict, *, config_hash_override: str | None = None):
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "results" / "phase2_pipeline"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = 20
    train_df = pd.DataFrame(
        {"feature_a": range(n), "isFraud": [0] * (n - 2) + [1, 1]},
        index=pd.Index(range(1, n + 1), name="TransactionID"),
    )
    val_df = pd.DataFrame(
        {"feature_a": range(5), "isFraud": [0, 0, 0, 0, 1]},
        index=pd.Index(range(n + 1, n + 6), name="TransactionID"),
    )
    test_df = pd.DataFrame(
        {"feature_a": range(5), "isFraud": [0, 0, 0, 1, 0]},
        index=pd.Index(range(n + 6, n + 11), name="TransactionID"),
    )
    train_df.to_pickle(processed_dir / "train.pkl")
    val_df.to_pickle(processed_dir / "val.pkl")
    test_df.to_pickle(processed_dir / "test.pkl")

    all_ids = list(train_df.index) + list(val_df.index) + list(test_df.index)
    assignments = pd.DataFrame(
        {
            "TransactionID": all_ids,
            "TransactionDT": [i * 100 for i in range(len(all_ids))],
            "split": ["train"] * n + ["val"] * 5 + ["test"] * 5,
            "cv_eval_fold": [None] * len(all_ids),
        }
    )
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)

    boundaries = SplitBoundaries(
        dt_min=0, dt_max=3000, train_val_boundary=1900, val_test_boundary=2400, cv_boundaries=(1000, 1900)
    )
    (output_dir / "split_boundaries.json").write_text(json.dumps(boundaries.to_dict()), encoding="utf-8")

    config_hash = config_hash_override or data_provenance.compute_config_hash(config)
    (output_dir / "provenance.json").write_text(
        json.dumps({"config_hash": config_hash, "timestamp_utc": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    return boundaries


def _write_config(tmp_path, config: dict) -> str:
    config_dir = tmp_path / "configs" / "phase2"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(config_path)


@pytest.fixture
def fake_config():
    return {
        "phase": 2,
        "seed": 42,
        "data": {"raw_dir": "data/raw", "transaction_file": "t.csv", "identity_file": "i.csv", "expected_sha256": {}},
        "load": {"chunksize": 100},
        "split": {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 2},
        "preprocess": {"exclude_columns": [], "passthrough_columns": []},
        "output": {"dir": "results/phase2_pipeline", "cache_processed_data": True, "processed_dir": "data/processed"},
    }


@pytest.fixture
def patched_project_root(tmp_path, monkeypatch):
    import src.config as config_module
    import src.train.data as train_data_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(train_data_module, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_load_phase2_features_uses_fresh_cache_without_rerunning_pipeline(
    tmp_path, fake_config, patched_project_root, monkeypatch
):
    _write_fake_phase2_output(tmp_path, fake_config)
    config_path = _write_config(tmp_path, fake_config)

    import src.train.data as train_data_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("pipeline.run() should not be called when the cache is fresh")

    monkeypatch.setattr(train_data_module.data_pipeline, "run", _fail_if_called)

    features = train_data_module.load_phase2_features(config_path)
    assert len(features.X_train) == 20
    assert len(features.X_val) == 5
    assert len(features.X_test) == 5


def test_load_phase2_features_reruns_pipeline_when_config_hash_stale(
    tmp_path, fake_config, patched_project_root, monkeypatch
):
    # Recorded provenance has a config_hash that does NOT match fake_config.
    _write_fake_phase2_output(tmp_path, fake_config, config_hash_override="stale_hash_value")
    config_path = _write_config(tmp_path, fake_config)

    import src.train.data as train_data_module

    call_count = {"n": 0}

    def _rerun_stub(cfg_path):
        call_count["n"] += 1
        # A real run would regenerate the cache with the correct hash;
        # simulate that so the subsequent pickle-read succeeds.
        _write_fake_phase2_output(tmp_path, fake_config)

    monkeypatch.setattr(train_data_module.data_pipeline, "run", _rerun_stub)

    train_data_module.load_phase2_features(config_path)
    assert call_count["n"] == 1


def test_load_phase2_features_reruns_pipeline_when_cache_files_missing(
    tmp_path, fake_config, patched_project_root, monkeypatch
):
    config_path = _write_config(tmp_path, fake_config)  # no cache written at all

    import src.train.data as train_data_module

    call_count = {"n": 0}

    def _rerun_stub(cfg_path):
        call_count["n"] += 1
        _write_fake_phase2_output(tmp_path, fake_config)

    monkeypatch.setattr(train_data_module.data_pipeline, "run", _rerun_stub)

    train_data_module.load_phase2_features(config_path)
    assert call_count["n"] == 1


def test_load_phase2_features_train_dt_aligned_to_x_train_order(
    tmp_path, fake_config, patched_project_root
):
    _write_fake_phase2_output(tmp_path, fake_config)
    config_path = _write_config(tmp_path, fake_config)

    import src.train.data as train_data_module

    features = train_data_module.load_phase2_features(config_path)
    assert list(features.train_dt.index) == list(features.X_train.index)


def test_load_phase2_features_boundaries_reconstructed_correctly(
    tmp_path, fake_config, patched_project_root
):
    original_boundaries = _write_fake_phase2_output(tmp_path, fake_config)
    config_path = _write_config(tmp_path, fake_config)

    import src.train.data as train_data_module

    features = train_data_module.load_phase2_features(config_path)
    assert features.boundaries == original_boundaries


def test_load_phase2_features_isfraud_excluded_from_x(tmp_path, fake_config, patched_project_root):
    _write_fake_phase2_output(tmp_path, fake_config)
    config_path = _write_config(tmp_path, fake_config)

    import src.train.data as train_data_module

    features = train_data_module.load_phase2_features(config_path)
    assert "isFraud" not in features.X_train.columns
    assert "isFraud" not in features.X_val.columns
    assert "isFraud" not in features.X_test.columns
