import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.data.load import (
    compute_file_digest,
    infer_dtype_map,
    iter_chunks,
    load_full,
    read_header,
)

LOAD_CONFIG = {"chunksize": 3, "downcast_floats": True, "categorical_as_category": True}


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7],
            "TransactionDT": [0, 3600, 7200, 10800, 14400, 18000, 21600],
            "isFraud": [0, 1, 0, 0, 1, 0, 0],
            "TransactionAmt": [10.5, 20.0, 5.25, 100.0, 7.5, 12.0, 99.9],
            "ProductCD": ["W", "C", "W", "R", "W", "C", "W"],
            "card4": ["visa", "mastercard", "visa", None, "visa", "mastercard", "visa"],
            "V1": [1.0, None, 1.0, None, 1.0, None, 1.0],
            "V2": [None, 2.0, None, 2.0, None, 2.0, None],
        }
    )
    path = tmp_path / "synthetic_transaction.csv"
    df.to_csv(path, index=False)
    return path


def test_read_header_returns_exact_columns(synthetic_csv):
    header = read_header(synthetic_csv)
    assert header == [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "V1",
        "V2",
    ]


def test_infer_dtype_map_classifies_correctly(synthetic_csv):
    dtype_map = infer_dtype_map(synthetic_csv, LOAD_CONFIG)
    assert dtype_map["TransactionID"] == "int64"
    assert dtype_map["TransactionDT"] == "int64"
    assert dtype_map["isFraud"] == "float32"
    assert dtype_map["TransactionAmt"] == "float32"
    assert dtype_map["ProductCD"] == "category"
    assert dtype_map["card4"] == "category"
    assert dtype_map["V1"] == "float32"
    assert dtype_map["V2"] == "float32"


def test_infer_dtype_map_respects_downcast_flag(synthetic_csv):
    config = dict(LOAD_CONFIG, downcast_floats=False)
    dtype_map = infer_dtype_map(synthetic_csv, config)
    assert dtype_map["TransactionAmt"] == "float64"


def test_infer_dtype_map_respects_category_flag(synthetic_csv):
    config = dict(LOAD_CONFIG, categorical_as_category=False)
    dtype_map = infer_dtype_map(synthetic_csv, config)
    assert dtype_map["ProductCD"] == "object"


def test_iter_chunks_covers_all_rows_with_correct_dtypes(synthetic_csv):
    dtype_map = infer_dtype_map(synthetic_csv, LOAD_CONFIG)
    chunks = list(iter_chunks(synthetic_csv, LOAD_CONFIG, dtype_map))
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 7
    # chunksize=3 over 7 rows -> 3 chunks (3, 3, 1)
    assert [len(c) for c in chunks] == [3, 3, 1]
    for chunk in chunks:
        assert str(chunk["TransactionID"].dtype) == "int64"
        assert str(chunk["TransactionAmt"].dtype) == "float32"
        assert str(chunk["ProductCD"].dtype) == "category"


def test_load_full_with_usecols_returns_only_requested_columns(synthetic_csv):
    dtype_map = infer_dtype_map(synthetic_csv, LOAD_CONFIG)
    df = load_full(synthetic_csv, dtype_map, usecols=["TransactionID", "TransactionDT"])
    assert list(df.columns) == ["TransactionID", "TransactionDT"]
    assert len(df) == 7


def test_load_full_without_usecols_returns_all_columns(synthetic_csv):
    dtype_map = infer_dtype_map(synthetic_csv, LOAD_CONFIG)
    df = load_full(synthetic_csv, dtype_map)
    assert len(df.columns) == 8
    assert len(df) == 7


def test_compute_file_digest_matches_direct_sha256(synthetic_csv):
    expected = hashlib.sha256(synthetic_csv.read_bytes()).hexdigest()
    assert compute_file_digest(synthetic_csv) == expected


def test_compute_file_digest_is_deterministic(synthetic_csv):
    assert compute_file_digest(synthetic_csv) == compute_file_digest(synthetic_csv)
