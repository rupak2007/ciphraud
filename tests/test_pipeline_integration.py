"""Phase 2 pipeline wiring test on synthetic data.

Exercises the full src.data.pipeline.run() path end-to-end against a small
synthetic dataset -- proves the modules (load, schema, split, preprocess,
leakage, provenance) are wired together correctly without needing the real
~650MB dataset. tests/test_pipeline_real_data.py is the real-data
counterpart, skipif-gated on data/raw/ being populated.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.pipeline import run


def _write_synthetic_dataset(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    n = 400
    rng_amt = [10.0 + (i % 37) * 3.3 for i in range(n)]
    transaction_df = pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "isFraud": [1 if i % 11 == 0 else 0 for i in range(n)],
            "TransactionDT": [i * 3600 for i in range(n)],  # 1 txn/hour, strictly increasing
            "TransactionAmt": rng_amt,
            "ProductCD": [["W", "C", "R", "H", "S"][i % 5] for i in range(n)],
            "card4": [["visa", "mastercard", None][i % 3] for i in range(n)],
            "V1": [1.0 if i % 4 else None for i in range(n)],
        }
    )
    transaction_df.to_csv(raw_dir / "train_transaction.csv", index=False)

    identity_ids = [i for i in range(1, n + 1) if i % 3 == 0]
    identity_df = pd.DataFrame(
        {
            "TransactionID": identity_ids,
            "DeviceType": [["mobile", "desktop"][i % 2] for i in range(len(identity_ids))],
        }
    )
    identity_df.to_csv(raw_dir / "train_identity.csv", index=False)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def synthetic_project(tmp_path: Path, monkeypatch) -> Path:
    """A self-contained fake project root: data/raw + configs/phase2/pipeline.yaml.

    Monkeypatches src.config.PROJECT_ROOT (used throughout the pipeline
    for path resolution) so the real project's data/results directories
    are never touched by this test.
    """
    raw_dir = tmp_path / "data" / "raw"
    _write_synthetic_dataset(raw_dir)

    tx_sha = _sha256(raw_dir / "train_transaction.csv")
    id_sha = _sha256(raw_dir / "train_identity.csv")

    config = {
        "phase": 2,
        "seed": 42,
        "data": {
            "raw_dir": "data/raw",
            "transaction_file": "train_transaction.csv",
            "identity_file": "train_identity.csv",
            "expected_sha256": {
                "train_transaction.csv": tx_sha,
                "train_identity.csv": id_sha,
            },
        },
        "load": {"chunksize": 100, "downcast_floats": True, "categorical_as_category": True},
        "split": {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 2},
        "preprocess": {
            "exclude_columns": ["TransactionID", "TransactionDT", "isFraud"],
            "passthrough_columns": ["has_identity"],
        },
        "output": {
            "dir": "results/phase2_pipeline",
            "cache_processed_data": True,
            "processed_dir": "data/processed",
        },
    }
    config_dir = tmp_path / "configs" / "phase2"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import src.config as config_module

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_module, "CONFIGS_DIR", tmp_path / "configs")

    import src.data.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "PROJECT_ROOT", tmp_path)

    return tmp_path


def test_pipeline_runs_end_to_end_on_synthetic_data(synthetic_project):
    result = run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    assert result["audit_report"]["all_passed"] is True


def test_pipeline_writes_all_expected_output_files(synthetic_project):
    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    output_dir = synthetic_project / "results" / "phase2_pipeline"
    for filename in (
        "split_assignments.csv",
        "split_boundaries.json",
        "leakage_audit_report.json",
        "preprocessing_summary.json",
        "provenance.json",
    ):
        assert (output_dir / filename).exists(), f"missing {filename}"


def test_pipeline_split_assignments_cover_every_row(synthetic_project):
    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    output_dir = synthetic_project / "results" / "phase2_pipeline"
    assignments = pd.read_csv(output_dir / "split_assignments.csv")
    assert len(assignments) == 400
    assert set(assignments["split"].unique()) <= {"train", "val", "test"}


def test_pipeline_feature_matrix_never_contains_banned_columns(synthetic_project):
    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    output_dir = synthetic_project / "results" / "phase2_pipeline"
    summary = json.loads((output_dir / "preprocessing_summary.json").read_text())
    for banned in ("TransactionID", "TransactionDT", "isFraud"):
        assert banned not in summary["feature_columns"]


def test_pipeline_caches_processed_data_when_enabled(synthetic_project):
    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    processed_dir = synthetic_project / "data" / "processed"
    assert (processed_dir / "train.pkl").exists()
    assert (processed_dir / "val.pkl").exists()
    assert (processed_dir / "test.pkl").exists()
    cached_train = pd.read_pickle(processed_dir / "train.pkl")
    assert "isFraud" in cached_train.columns


def test_pipeline_is_deterministic_across_runs(synthetic_project):
    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    output_dir = synthetic_project / "results" / "phase2_pipeline"
    first_assignments = pd.read_csv(output_dir / "split_assignments.csv")

    run(str(synthetic_project / "configs" / "phase2" / "pipeline.yaml"))
    second_assignments = pd.read_csv(output_dir / "split_assignments.csv")

    pd.testing.assert_frame_equal(first_assignments, second_assignments)


def test_pipeline_sha256_mismatch_raises(synthetic_project):
    from src.config import load_config

    config_path = synthetic_project / "configs" / "phase2" / "pipeline.yaml"
    config = load_config(str(config_path))
    config["data"]["expected_sha256"]["train_transaction.csv"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    from src.data.load import DataAcquisitionError

    with pytest.raises(DataAcquisitionError, match="SHA256 mismatch"):
        run(str(config_path))
