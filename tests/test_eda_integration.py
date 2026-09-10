"""End-to-end test of src.data.eda.run() against a small synthetic dataset.

This is distinct from the real-data skipif-gated integration test
(planned for tests/test_eda_real_data.py once the dataset is available):
it exercises the full Step A -> Step B -> Step C -> output-writing wiring
using tiny synthetic CSVs, so wiring bugs (missing config keys, argument
mismatches) are caught without needing the real ~650 MB download.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.eda import run

N_ROWS = 40


def _make_synthetic_transaction_csv(path: Path) -> None:
    dt = [i * 3600 for i in range(N_ROWS)]  # one row per hour, spans ~40 hours
    fraud = [1 if i % 7 == 0 else 0 for i in range(N_ROWS)]
    df = pd.DataFrame(
        {
            "TransactionID": list(range(1, N_ROWS + 1)),
            "TransactionDT": dt,
            "isFraud": fraud,
            "TransactionAmt": [10.0 + i for i in range(N_ROWS)],
            "ProductCD": ["W" if i % 2 == 0 else "C" for i in range(N_ROWS)],
            "card4": ["visa" if i % 3 else "mastercard" for i in range(N_ROWS)],
            "V1": [1.0 if i % 2 == 0 else None for i in range(N_ROWS)],
            "V2": [2.0 if i % 2 == 0 else None for i in range(N_ROWS)],  # matches V1's null mask
            "V3": [3.0 if i % 4 == 0 else None for i in range(N_ROWS)],  # different mask
        }
    )
    df.to_csv(path, index=False)


def _make_synthetic_identity_csv(path: Path) -> None:
    # Only a subset of TransactionIDs have an identity row.
    df = pd.DataFrame({"TransactionID": list(range(1, N_ROWS + 1, 2))})
    df.to_csv(path, index=False)


def test_run_end_to_end_on_synthetic_data(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "results"

    _make_synthetic_transaction_csv(raw_dir / "train_transaction.csv")
    _make_synthetic_identity_csv(raw_dir / "train_identity.csv")

    config = {
        "phase": 1,
        "seed": 42,
        "data": {
            "raw_dir": str(raw_dir),
            "transaction_file": "train_transaction.csv",
            "identity_file": "train_identity.csv",
            "expected_sha256": {"train_transaction.csv": "", "train_identity.csv": ""},
        },
        "load": {"chunksize": 10, "downcast_floats": True, "categorical_as_category": True},
        "eda": {"time_bucket_seconds": 86400, "candidate_split_quantiles": [0.6, 0.7, 0.8]},
        "output": {"dir": str(output_dir)},
    }
    config_path = tmp_path / "eda.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    run(str(config_path))

    expected_files = [
        "column_inventory.csv",
        "missingness_global.csv",
        "missingness_by_time_bucket.csv",
        "v_null_mask_blocks.csv",
        "class_balance_by_time.csv",
        "numeric_ranges.csv",
        "oov_rates.csv",
        "cardinality.csv",
        "candidate_split_boundaries.csv",
        "transactiondt_semantics.json",
        "dataset_summary.json",
        "provenance.json",
        "fraud_rate_over_time.png",
        "volume_over_time.png",
        "hour_of_day_histogram.png",
        "transaction_amt_distribution.png",
    ]
    for filename in expected_files:
        f = output_dir / filename
        assert f.exists(), f"missing expected output: {filename}"
        assert f.stat().st_size > 0, f"empty output file: {filename}"

    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["n_transaction_rows"] == N_ROWS
    assert summary["n_identity_rows"] == len(range(1, N_ROWS + 1, 2))

    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["seed"] == 42
    assert "git_commit" in provenance
    assert provenance["raw_file_sha256"]["train_transaction.csv"]  # non-empty digest recorded


def test_run_is_deterministic_across_repeated_runs(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir = tmp_path / "results"
    _make_synthetic_transaction_csv(raw_dir / "train_transaction.csv")
    _make_synthetic_identity_csv(raw_dir / "train_identity.csv")

    config = {
        "phase": 1,
        "seed": 42,
        "data": {
            "raw_dir": str(raw_dir),
            "transaction_file": "train_transaction.csv",
            "identity_file": "train_identity.csv",
            "expected_sha256": {"train_transaction.csv": "", "train_identity.csv": ""},
        },
        "load": {"chunksize": 10, "downcast_floats": True, "categorical_as_category": True},
        "eda": {"time_bucket_seconds": 86400, "candidate_split_quantiles": [0.6, 0.7, 0.8]},
        "output": {"dir": str(output_dir)},
    }
    config_path = tmp_path / "eda.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    run(str(config_path))
    first = (output_dir / "missingness_global.csv").read_text(encoding="utf-8")
    run(str(config_path))
    second = (output_dir / "missingness_global.csv").read_text(encoding="utf-8")
    assert first == second
