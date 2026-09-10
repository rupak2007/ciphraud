"""Memory-aware loading for the IEEE-CIS Fraud Detection CSVs.

The host has 7.95 GB RAM (see docs/environment.md) and train_transaction.csv
alone is ~590,540 rows x 394 columns. Measured on this machine (pandas
3.0.5, no pyarrow): `category` cuts a text column's memory by ~57x versus
`object`/`str`, and `float32` halves `float64`. pandas 3.0's new `str`
dtype gives *zero* memory benefit over `object` here (pyarrow-backed
string storage isn't installed) -- so `category` + `float32` are the only
real levers, not the string dtype itself.

Chunked reading is the primary path for full-file aggregation (bounded,
exact memory use); `load_full` is an explicit, measured, optional second
pass -- never assume a full load's footprint, report it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

# TransactionID (primary key) and TransactionDT (the time-bucketing /
# ordering key) are kept as exact int64 rather than downcast to float32.
# This is a deliberate choice, not an oversight: Phase 1's time-bucket and
# hour-of-day arithmetic needs exact integer semantics, and both columns
# are expected to be fully populated (TransactionID by construction,
# TransactionDT because it is the dataset's sort key). If either turns out
# to contain a null in a later chunk than the sample used to infer dtypes,
# pandas' C parser will raise a clear "integer column has NA values"
# error -- that is treated as a genuine, loud data-quality finding to
# investigate, not something to silently coerce around.
_EXACT_INT_COLUMNS = ("TransactionID", "TransactionDT")


def read_header(path: Path) -> list[str]:
    """Read only the column names, without loading any rows."""
    return list(pd.read_csv(path, nrows=0).columns)


def infer_dtype_map(path: Path, load_config: dict[str, Any]) -> dict[str, str]:
    """Infer a target dtype for every column from a sample of real rows.

    Deliberately empirical rather than a hardcoded IEEE-CIS column-name
    table: a sample of `load_config["chunksize"]` rows is read with
    pandas' default inference, and each column's *observed* dtype (not an
    assumed one) decides whether it becomes float32/float64 (numeric) or
    category (everything else). See module docstring for why this, not
    the sample's exact dtype, is downcast-targeted.
    """
    sample = pd.read_csv(path, nrows=load_config["chunksize"])
    downcast_floats = load_config.get("downcast_floats", True)
    use_category = load_config.get("categorical_as_category", True)

    dtype_map: dict[str, str] = {}
    for col in sample.columns:
        if col in _EXACT_INT_COLUMNS:
            dtype_map[col] = "int64"
        elif pd.api.types.is_numeric_dtype(sample[col]):
            dtype_map[col] = "float32" if downcast_floats else "float64"
        else:
            dtype_map[col] = "category" if use_category else "object"
    return dtype_map


def iter_chunks(
    path: Path, load_config: dict[str, Any], dtype_map: dict[str, str]
) -> Iterator[pd.DataFrame]:
    """Stream the file in bounded-memory chunks using a fixed dtype map.

    This is the primary path for full-file aggregation (missingness,
    class balance, cardinality, etc.) -- exact, and memory use is bounded
    by `chunksize` regardless of file size.
    """
    reader = pd.read_csv(
        path,
        dtype=dtype_map,
        chunksize=load_config["chunksize"],
    )
    yield from reader


def load_full(
    path: Path,
    dtype_map: dict[str, str],
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load the file in one pass with a fixed dtype map.

    An explicit, optional second pass -- not the default. Callers must
    measure and report `df.memory_usage(deep=True).sum()` themselves
    rather than assume a footprint; this project has no approved tool for
    measuring process peak RSS on Windows (psutil isn't installed; Phase 7
    owns that decision), so only the resulting DataFrame's own footprint
    is a claim this function can honestly support.
    """
    effective_dtype_map = (
        dtype_map if usecols is None else {k: v for k, v in dtype_map.items() if k in usecols}
    )
    return pd.read_csv(path, dtype=effective_dtype_map, usecols=usecols)


def compute_file_digest(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA256 of a file, for download-integrity verification."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()
