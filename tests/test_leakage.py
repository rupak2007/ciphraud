"""Leakage-audit tests.

Every test in this file that ends in `_raises` constructs a DELIBERATELY
leaky scenario and asserts the audit catches it -- these are the tests
CLAUDE.md Sec.11 and instructions.md's Testing Rules require ("unit tests
asserting no test-set or future-window information leaks"). A leakage
check with no failing-case test is unverified; every check below has one.
"""

import pandas as pd
import pytest

from src.data.leakage import (
    LeakageError,
    check_banned_columns_absent,
    check_cv_folds_never_reach_test,
    check_fitted_statistic_uses_only_allowed_index,
    check_no_duplicate_dt_straddles_boundary,
    check_split_is_temporally_ordered,
    check_split_partitions_are_disjoint_and_complete,
    run_leakage_audit,
)
from src.data.preprocess import TrainScopedFrequencyEncoder, TrainScopedMedianImputer


# ---------------------------------------------------------------------------
# check_split_partitions_are_disjoint_and_complete
# ---------------------------------------------------------------------------


def test_split_partitions_passes_on_clean_labels():
    labels = pd.Series(["train", "train", "val", "test"])
    check_split_partitions_are_disjoint_and_complete(labels)  # must not raise


def test_split_partitions_raises_on_unexpected_label():
    labels = pd.Series(["train", "val", "bogus"])
    with pytest.raises(LeakageError, match="unexpected"):
        check_split_partitions_are_disjoint_and_complete(labels)


def test_split_partitions_raises_on_null_label():
    labels = pd.Series(["train", None, "val"])
    with pytest.raises(LeakageError, match="no split label"):
        check_split_partitions_are_disjoint_and_complete(labels)


# ---------------------------------------------------------------------------
# check_split_is_temporally_ordered
# ---------------------------------------------------------------------------


def test_temporally_ordered_passes_on_clean_split():
    dt = pd.Series([1, 2, 3, 10, 11, 20, 21])
    labels = pd.Series(["train", "train", "train", "val", "val", "test", "test"])
    check_split_is_temporally_ordered(dt, labels)  # must not raise


def test_temporally_ordered_raises_when_a_future_row_is_labeled_train():
    """The canonical leakage scenario: a row from the future accidentally
    lands in 'train' (e.g. a bug that assigns by random shuffle instead of
    by DT threshold)."""
    dt = pd.Series([1, 2, 3, 10, 11, 999])  # last row is far in the future...
    labels = pd.Series(["train", "train", "train", "val", "val", "train"])  # ...but labeled train
    with pytest.raises(LeakageError, match="not temporally ordered"):
        check_split_is_temporally_ordered(dt, labels)


def test_temporally_ordered_raises_when_val_and_test_overlap():
    dt = pd.Series([1, 2, 50, 10, 11])
    labels = pd.Series(["train", "train", "val", "val", "test"])  # DT=50 val row is after DT=11 test row
    with pytest.raises(LeakageError, match="not temporally ordered"):
        check_split_is_temporally_ordered(dt, labels)


# ---------------------------------------------------------------------------
# check_no_duplicate_dt_straddles_boundary
# ---------------------------------------------------------------------------


def test_no_duplicate_dt_straddle_passes_when_ties_stay_together():
    dt = pd.Series([1, 1, 1, 5, 5, 9])
    labels = pd.Series(["train", "train", "train", "val", "val", "test"])
    check_no_duplicate_dt_straddles_boundary(dt, labels)  # must not raise


def test_no_duplicate_dt_straddle_raises_when_a_tie_is_split():
    dt = pd.Series([1, 1, 1, 5])
    labels = pd.Series(["train", "train", "val", "test"])  # DT=1 split across train/val
    with pytest.raises(LeakageError, match="more than one split"):
        check_no_duplicate_dt_straddles_boundary(dt, labels)


# ---------------------------------------------------------------------------
# check_cv_folds_never_reach_test
# ---------------------------------------------------------------------------


def test_cv_folds_never_reach_test_passes_when_boundaries_precede_test():
    dt = pd.Series([1, 2, 3, 4, 5, 100, 101])
    labels = pd.Series(["train", "train", "train", "train", "train", "test", "test"])
    check_cv_folds_never_reach_test(dt, labels, cv_boundaries=[2, 4])


def test_cv_folds_never_reach_test_raises_when_a_boundary_reaches_test():
    dt = pd.Series([1, 2, 3, 100, 101])
    labels = pd.Series(["train", "train", "train", "test", "test"])
    with pytest.raises(LeakageError, match="reaches into the primary test partition"):
        check_cv_folds_never_reach_test(dt, labels, cv_boundaries=[2, 100])


# ---------------------------------------------------------------------------
# check_fitted_statistic_uses_only_allowed_index
# ---------------------------------------------------------------------------


def test_fitted_statistic_scope_passes_when_fit_on_train_only():
    train_index = pd.Index([0, 1, 2, 3])
    check_fitted_statistic_uses_only_allowed_index(pd.Index([0, 1, 2]), train_index, "test_stat")


def test_fitted_statistic_scope_raises_when_fit_includes_val_rows():
    """The canonical encoder-leakage scenario: frequency table computed
    over train+val combined instead of train alone."""
    train_index = pd.Index([0, 1, 2])
    fit_index_including_val = pd.Index([0, 1, 2, 3, 4])  # 3, 4 are val rows
    with pytest.raises(LeakageError, match="outside the allowed train-only index"):
        check_fitted_statistic_uses_only_allowed_index(fit_index_including_val, train_index, "test_stat")


def test_real_encoder_fit_on_val_contaminated_data_is_caught():
    """End-to-end version of the above using the actual encoder class:
    fit a TrainScopedFrequencyEncoder on train+val combined (a realistic
    implementation mistake) and confirm the audit catches it."""
    train_df = pd.DataFrame({"cat": ["a", "b", "a"]}, index=[0, 1, 2])
    val_df = pd.DataFrame({"cat": ["b", "c"]}, index=[3, 4])
    contaminated = pd.concat([train_df, val_df])

    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(contaminated)  # BUG: fit on train+val

    with pytest.raises(LeakageError):
        check_fitted_statistic_uses_only_allowed_index(encoder.fit_index_, train_df.index, "cat_encoder")


def test_real_imputer_fit_on_val_contaminated_data_is_caught():
    train_df = pd.DataFrame({"amt": [10.0, 20.0]}, index=[0, 1])
    val_df = pd.DataFrame({"amt": [999.0]}, index=[2])
    contaminated = pd.concat([train_df, val_df])

    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(contaminated)  # BUG: fit on train+val

    with pytest.raises(LeakageError):
        check_fitted_statistic_uses_only_allowed_index(imputer.fit_index_, train_df.index, "amt_imputer")


# ---------------------------------------------------------------------------
# check_banned_columns_absent
# ---------------------------------------------------------------------------


def test_banned_columns_absent_passes_when_clean():
    check_banned_columns_absent(["amt_freq", "has_identity"], banned=["TransactionID", "isFraud"])


def test_banned_columns_absent_raises_when_transaction_id_present():
    with pytest.raises(LeakageError, match="TransactionID"):
        check_banned_columns_absent(
            ["amt_freq", "TransactionID"], banned=["TransactionID", "TransactionDT", "isFraud"]
        )


# ---------------------------------------------------------------------------
# run_leakage_audit -- full integration of all checks
# ---------------------------------------------------------------------------


def _clean_scenario():
    dt = pd.Series(range(0, 100))
    labels = pd.Series(["train"] * 60 + ["val"] * 20 + ["test"] * 20)
    train_index = pd.RangeIndex(60)
    encoder = TrainScopedFrequencyEncoder(columns=[]).fit(pd.DataFrame(index=train_index))
    imputer = TrainScopedMedianImputer(columns=[]).fit(pd.DataFrame(index=train_index))
    return dt, labels, train_index, encoder, imputer


def test_run_leakage_audit_passes_on_clean_scenario():
    dt, labels, train_index, encoder, imputer = _clean_scenario()
    report = run_leakage_audit(
        dt=dt,
        split_labels=labels,
        cv_boundaries=[10, 30, 50],
        feature_columns=["amt_freq", "has_identity"],
        banned_columns=["TransactionID", "TransactionDT", "isFraud"],
        fitted_statistics={"encoder": encoder, "imputer": imputer},
        train_index=train_index,
    )
    assert report["all_passed"] is True
    assert report["n_checks"] > 0


def test_run_leakage_audit_raises_on_banned_column_present():
    dt, labels, train_index, encoder, imputer = _clean_scenario()
    with pytest.raises(LeakageError):
        run_leakage_audit(
            dt=dt,
            split_labels=labels,
            cv_boundaries=[10, 30, 50],
            feature_columns=["amt_freq", "TransactionID"],
            banned_columns=["TransactionID", "TransactionDT", "isFraud"],
            fitted_statistics={"encoder": encoder, "imputer": imputer},
            train_index=train_index,
        )


def test_run_leakage_audit_raises_on_contaminated_fitted_statistic():
    dt, labels, train_index, _, imputer = _clean_scenario()
    contaminated_encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(
        pd.DataFrame({"cat": ["a"]}, index=pd.RangeIndex(60, 65))  # includes val rows 60-64
    )
    with pytest.raises(LeakageError):
        run_leakage_audit(
            dt=dt,
            split_labels=labels,
            cv_boundaries=[10, 30, 50],
            feature_columns=["cat_freq"],
            banned_columns=["TransactionID", "TransactionDT", "isFraud"],
            fitted_statistics={"encoder": contaminated_encoder, "imputer": imputer},
            train_index=train_index,
        )
