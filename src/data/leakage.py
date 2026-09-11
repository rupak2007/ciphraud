"""Leakage audit checks for the Phase 2 split/preprocessing pipeline.

CLAUDE.md Sec.6 ("Temporal leakage is a critical correctness issue") and
instructions.md's Data-Leakage Rules require every split and every
engineered feature to use only information available strictly before the
window it's applied to, and require any leakage found to be fixed and
documented, not silently patched. Every check in this module is designed
to FAIL LOUDLY: each raises `LeakageError` immediately on violation rather
than warning or returning a boolean, matching the `SchemaValidationError`
pattern already established in `src/data/schema.py`.

`run_leakage_audit` runs every check in a fixed order and returns a report
dict only if *all* checks pass; a failing check propagates its exception
and the pipeline run crashes -- that is the intended behavior, not a bug
to catch and hide.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


class LeakageError(Exception):
    """Raised when a leakage-safety invariant is violated."""


# ---------------------------------------------------------------------------
# Split-level checks
# ---------------------------------------------------------------------------


def check_split_partitions_are_disjoint_and_complete(
    split_labels: pd.Series, expected_labels: Iterable[str] = ("train", "val", "test")
) -> None:
    """Every row has exactly one of the expected labels; nothing is dropped or duplicated."""
    if split_labels.isna().any():
        raise LeakageError(f"{int(split_labels.isna().sum())} row(s) have no split label")
    unexpected = set(split_labels.unique()) - set(expected_labels)
    if unexpected:
        raise LeakageError(f"Split contains unexpected label(s): {sorted(unexpected)}")


def check_split_is_temporally_ordered(dt: pd.Series, split_labels: pd.Series) -> None:
    """max(train.DT) <= min(val.DT) <= ... <= min(test.DT): no partition reaches into the future of another."""
    train_dt = dt[split_labels == "train"]
    val_dt = dt[split_labels == "val"]
    test_dt = dt[split_labels == "test"]
    if len(train_dt) and len(val_dt) and train_dt.max() > val_dt.min():
        raise LeakageError(
            f"Split is not temporally ordered: max(train.DT)={train_dt.max()} > "
            f"min(val.DT)={val_dt.min()}"
        )
    if len(val_dt) and len(test_dt) and val_dt.max() > test_dt.min():
        raise LeakageError(
            f"Split is not temporally ordered: max(val.DT)={val_dt.max()} > "
            f"min(test.DT)={test_dt.min()}"
        )
    if len(train_dt) and len(test_dt) and train_dt.max() > test_dt.min():
        raise LeakageError(
            f"Split is not temporally ordered: max(train.DT)={train_dt.max()} > "
            f"min(test.DT)={test_dt.min()}"
        )


def check_no_duplicate_dt_straddles_boundary(dt: pd.Series, split_labels: pd.Series) -> None:
    """No single TransactionDT value appears in more than one split partition.

    docs/eda.md Sec.2 measured 17,191 adjacent duplicate TransactionDT
    values; a threshold-based split (src/data/split.py) makes this safe by
    construction, but this check verifies it directly on the actual
    assignment rather than trusting the construction to never regress.
    """
    grouped = pd.DataFrame({"dt": dt.to_numpy(), "split": split_labels.to_numpy()})
    n_labels_per_dt = grouped.groupby("dt")["split"].nunique()
    straddling = n_labels_per_dt[n_labels_per_dt > 1]
    if len(straddling):
        raise LeakageError(
            f"{len(straddling)} TransactionDT value(s) appear in more than one split "
            f"partition -- a split boundary landed inside a tied-DT block "
            f"(e.g. DT={straddling.index[0]})."
        )


def check_cv_folds_never_reach_test(
    dt: pd.Series, split_labels: pd.Series, cv_boundaries: Iterable[int]
) -> None:
    """No expanding-window CV boundary may reach into the primary test partition."""
    test_dt = dt[split_labels == "test"]
    if not len(test_dt):
        return
    test_dt_min = test_dt.min()
    for boundary in cv_boundaries:
        if boundary >= test_dt_min:
            raise LeakageError(
                f"CV fold boundary {boundary} reaches into the primary test partition "
                f"(test starts at DT={test_dt_min}) -- CV must never touch test."
            )


# ---------------------------------------------------------------------------
# Fitted-statistic checks
# ---------------------------------------------------------------------------


def check_fitted_statistic_uses_only_allowed_index(
    fit_index: pd.Index, allowed_index: pd.Index, name: str
) -> None:
    """Verify a fitted encoder/imputer's recorded fit-index is a subset of what's allowed.

    Every `TrainScopedFrequencyEncoder`/`TrainScopedMedianImputer` in
    `src/data/preprocess.py` records exactly which rows it was fit on
    (`fit_index_`). This check is the leakage-audit's teeth: it doesn't
    trust that the caller passed the right DataFrame to `.fit()`, it
    verifies the fitted object's own record against the split's allowed
    index.
    """
    disallowed = set(fit_index) - set(allowed_index)
    if disallowed:
        sample = sorted(disallowed)[:5]
        raise LeakageError(
            f"{name} was fit using {len(disallowed)} row(s) outside the allowed "
            f"train-only index (e.g. {sample}) -- this is a direct temporal leak."
        )


# ---------------------------------------------------------------------------
# Feature-matrix checks
# ---------------------------------------------------------------------------


def check_banned_columns_absent(columns: Iterable[str], banned: Iterable[str]) -> None:
    """Assert none of `banned` (e.g. TransactionID, isFraud) appear in a feature matrix.

    TransactionID is banned per project mandate: docs/eda.md Sec.2 measured
    Spearman(TransactionID, TransactionDT) ~= 1, making it a near-perfect
    time-order proxy.
    """
    present = set(columns) & set(banned)
    if present:
        raise LeakageError(f"Banned column(s) present in feature matrix: {sorted(present)}")


# ---------------------------------------------------------------------------
# Top-level audit
# ---------------------------------------------------------------------------


def run_leakage_audit(
    *,
    dt: pd.Series,
    split_labels: pd.Series,
    cv_boundaries: Iterable[int],
    feature_columns: Iterable[str],
    banned_columns: Iterable[str],
    fitted_statistics: dict[str, Any],
    train_index: pd.Index,
) -> dict[str, Any]:
    """Run every leakage check in sequence; return a report dict iff all pass.

    `fitted_statistics` maps a human-readable name to any object exposing
    a `.fit_index_` attribute (both preprocessing classes in
    `src/data/preprocess.py` do). A failing check raises and this function
    never returns -- the caller (src/data/pipeline.py) lets that exception
    crash the run, per CLAUDE.md's "fail loudly" requirement.
    """
    checks_run: list[str] = []

    check_split_partitions_are_disjoint_and_complete(split_labels)
    checks_run.append("split_partitions_are_disjoint_and_complete")

    check_split_is_temporally_ordered(dt, split_labels)
    checks_run.append("split_is_temporally_ordered")

    check_no_duplicate_dt_straddles_boundary(dt, split_labels)
    checks_run.append("no_duplicate_dt_straddles_boundary")

    check_cv_folds_never_reach_test(dt, split_labels, cv_boundaries)
    checks_run.append("cv_folds_never_reach_test")

    check_banned_columns_absent(feature_columns, banned_columns)
    checks_run.append("banned_columns_absent")

    for name, fitted in fitted_statistics.items():
        check_fitted_statistic_uses_only_allowed_index(fitted.fit_index_, train_index, name)
        checks_run.append(f"fitted_statistic_scope[{name}]")

    return {
        "all_passed": True,
        "checks_run": checks_run,
        "n_checks": len(checks_run),
    }
