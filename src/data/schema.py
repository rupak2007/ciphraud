"""Schema definitions and validators for the IEEE-CIS Fraud Detection dataset.

Phase 1 deliberately does NOT hardcode the full ~394-column inventory as
ground truth. Only a small set of columns this project structurally
depends on (`REQUIRED_TRANSACTION_COLUMNS` / `REQUIRED_IDENTITY_COLUMNS`)
is asserted; the actual header is always read from the file and diffed
against expectations (see `diff_columns`). Treating a guessed full
inventory as truth would violate CLAUDE.md Sec.17 (measure, don't guess).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
)

REQUIRED_IDENTITY_COLUMNS: tuple[str, ...] = ("TransactionID",)

# id_01..id_38 appear hyphenated ("id-01") in some IEEE-CIS distributions
# and underscored ("id_01") in others (a well-known Kaggle quirk) -- match
# both rather than assuming one.
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("core", re.compile(r"^(TransactionID|isFraud|TransactionDT|TransactionAmt|ProductCD)$")),
    ("card", re.compile(r"^card\d+$")),
    ("addr", re.compile(r"^addr\d+$")),
    ("dist", re.compile(r"^dist\d+$")),
    ("email", re.compile(r"^(P|R)_emaildomain$")),
    ("C", re.compile(r"^C\d+$")),
    ("D", re.compile(r"^D\d+$")),
    ("M", re.compile(r"^M\d+$")),
    ("V", re.compile(r"^V\d+$")),
    ("id", re.compile(r"^id[_-]\d+$")),
    ("device", re.compile(r"^(DeviceType|DeviceInfo)$")),
)


def classify_column(name: str) -> str:
    """Map a raw column name to its IEEE-CIS family.

    Returns "unknown" rather than raising for anything that doesn't match
    a known naming pattern -- the actual header may contain columns this
    project hasn't accounted for, and that must surface in the
    actual-vs-expected column diff, not be swallowed by an exception.
    """
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.match(name):
            return family
    return "unknown"


class SchemaValidationError(Exception):
    """Raised when the dataset violates a structural invariant this project depends on."""


def validate_required_columns(
    columns: Iterable[str], required: Iterable[str], table_name: str
) -> None:
    missing = set(required) - set(columns)
    if missing:
        raise SchemaValidationError(
            f"{table_name}: missing required columns: {sorted(missing)}"
        )


def validate_unique_transaction_id(df: pd.DataFrame, table_name: str) -> None:
    duplicated = df["TransactionID"].duplicated()
    if duplicated.any():
        raise SchemaValidationError(
            f"{table_name}: TransactionID is not unique "
            f"({int(duplicated.sum())} duplicate rows)"
        )


def validate_identity_subset(transaction_ids: pd.Series, identity_ids: pd.Series) -> None:
    extra = set(identity_ids) - set(transaction_ids)
    if extra:
        sample = sorted(extra)[:5]
        raise SchemaValidationError(
            f"identity.TransactionID contains {len(extra)} IDs not present in "
            f"transaction.TransactionID (e.g. {sample})"
        )


def validate_is_fraud(series: pd.Series) -> None:
    if series.isna().any():
        raise SchemaValidationError(
            f"isFraud contains {int(series.isna().sum())} null values"
        )
    invalid = ~series.isin([0, 1])
    if invalid.any():
        raise SchemaValidationError(
            f"isFraud contains values outside {{0, 1}}: "
            f"{sorted(series[invalid].unique())[:5]}"
        )


def validate_transaction_amt(series: pd.Series) -> None:
    if series.isna().any():
        raise SchemaValidationError(
            f"TransactionAmt contains {int(series.isna().sum())} null values"
        )
    non_positive = series <= 0
    if non_positive.any():
        raise SchemaValidationError(
            f"TransactionAmt contains {int(non_positive.sum())} non-positive values"
        )


def diff_columns(actual: Iterable[str], expected: Iterable[str]) -> dict[str, list[str]]:
    """Report the gap between an actual header and an expected column set."""
    actual_set, expected_set = set(actual), set(expected)
    return {
        "missing_from_actual": sorted(expected_set - actual_set),
        "unexpected_in_actual": sorted(actual_set - expected_set),
    }


def validate_transaction_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df.columns, REQUIRED_TRANSACTION_COLUMNS, "train_transaction")
    validate_unique_transaction_id(df, "train_transaction")
    validate_is_fraud(df["isFraud"])
    validate_transaction_amt(df["TransactionAmt"])


def validate_identity_dataframe(df: pd.DataFrame) -> None:
    validate_required_columns(df.columns, REQUIRED_IDENTITY_COLUMNS, "train_identity")


def validate_identity_against_transaction(
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame
) -> None:
    validate_identity_subset(transaction_df["TransactionID"], identity_df["TransactionID"])
