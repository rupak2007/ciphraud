import pandas as pd
import pytest

from src.data.schema import (
    REQUIRED_IDENTITY_COLUMNS,
    REQUIRED_TRANSACTION_COLUMNS,
    SchemaValidationError,
    classify_column,
    diff_columns,
    validate_identity_subset,
    validate_is_fraud,
    validate_required_columns,
    validate_transaction_amt,
    validate_unique_transaction_id,
)


@pytest.mark.parametrize(
    "name,expected_family",
    [
        ("TransactionID", "core"),
        ("isFraud", "core"),
        ("TransactionDT", "core"),
        ("TransactionAmt", "core"),
        ("ProductCD", "core"),
        ("card1", "card"),
        ("card6", "card"),
        ("addr1", "addr"),
        ("addr2", "addr"),
        ("dist1", "dist"),
        ("dist2", "dist"),
        ("P_emaildomain", "email"),
        ("R_emaildomain", "email"),
        ("C1", "C"),
        ("C14", "C"),
        ("D1", "D"),
        ("D15", "D"),
        ("M1", "M"),
        ("M9", "M"),
        ("V1", "V"),
        ("V339", "V"),
        ("id_01", "id"),
        ("id-01", "id"),
        ("id_38", "id"),
        ("DeviceType", "device"),
        ("DeviceInfo", "device"),
        ("SomeUnexpectedColumn", "unknown"),
    ],
)
def test_classify_column(name, expected_family):
    assert classify_column(name) == expected_family


def test_validate_required_columns_passes_when_present():
    validate_required_columns(REQUIRED_TRANSACTION_COLUMNS, REQUIRED_TRANSACTION_COLUMNS, "t")


def test_validate_required_columns_raises_on_missing():
    with pytest.raises(SchemaValidationError, match="missing required columns"):
        validate_required_columns(["TransactionID"], REQUIRED_TRANSACTION_COLUMNS, "train_transaction")


def test_validate_unique_transaction_id_raises_on_duplicate():
    df = pd.DataFrame({"TransactionID": [1, 2, 2, 3]})
    with pytest.raises(SchemaValidationError, match="not unique"):
        validate_unique_transaction_id(df, "train_transaction")


def test_validate_unique_transaction_id_passes_when_unique():
    df = pd.DataFrame({"TransactionID": [1, 2, 3]})
    validate_unique_transaction_id(df, "train_transaction")


def test_validate_identity_subset_raises_when_extra_ids():
    transaction_ids = pd.Series([1, 2, 3])
    identity_ids = pd.Series([1, 2, 99])
    with pytest.raises(SchemaValidationError, match="not present in"):
        validate_identity_subset(transaction_ids, identity_ids)


def test_validate_identity_subset_passes_when_proper_subset():
    transaction_ids = pd.Series([1, 2, 3, 4])
    identity_ids = pd.Series([1, 3])
    validate_identity_subset(transaction_ids, identity_ids)


def test_validate_is_fraud_raises_on_null():
    with pytest.raises(SchemaValidationError, match="null values"):
        validate_is_fraud(pd.Series([0, 1, None]))


def test_validate_is_fraud_raises_on_invalid_value():
    with pytest.raises(SchemaValidationError, match="outside"):
        validate_is_fraud(pd.Series([0, 1, 2]))


def test_validate_is_fraud_passes_on_binary():
    validate_is_fraud(pd.Series([0, 1, 0, 1, 1]))


def test_validate_transaction_amt_raises_on_null():
    with pytest.raises(SchemaValidationError, match="null values"):
        validate_transaction_amt(pd.Series([10.0, None, 5.0]))


def test_validate_transaction_amt_raises_on_non_positive():
    with pytest.raises(SchemaValidationError, match="non-positive"):
        validate_transaction_amt(pd.Series([10.0, 0.0, -5.0]))


def test_validate_transaction_amt_passes_on_positive():
    validate_transaction_amt(pd.Series([10.0, 0.01, 500.0]))


def test_diff_columns_reports_both_directions():
    diff = diff_columns(actual=["A", "B", "X"], expected=["A", "B", "C"])
    assert diff == {"missing_from_actual": ["C"], "unexpected_in_actual": ["X"]}


def test_diff_columns_empty_when_identical():
    diff = diff_columns(actual=["A", "B"], expected=["B", "A"])
    assert diff == {"missing_from_actual": [], "unexpected_in_actual": []}
