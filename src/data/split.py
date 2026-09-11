"""Time-based expanding-window split for the IEEE-CIS Fraud Detection dataset.

CLAUDE.md Sec.6 and architecture.md Sec.3 mandate a **time-based expanding-window**
train/val/test split by `TransactionDT`; random k-fold cross-validation is
prohibited. This module provides two related but distinct things:

1. **The primary split** (`assign_primary_split`): a single train/val/test
   partition used for the headline evaluation numbers in later phases. Test
   is touched exactly once, at final evaluation time.
2. **Expanding-window CV folds** (`iter_expanding_window_folds`): a
   walk-forward generalization of the primary split's "train" + "val"
   region only -- each fold's train window strictly grows and its eval
   window immediately follows, never overlapping and never reaching into
   the primary test region. This gives later phases (seed-stability
   checks, threshold tuning, model selection) a genuine temporal
   cross-validation strategy without ever touching the final test set or
   resorting to random k-fold.

Both use the same boundary-computation methodology Phase 1 established in
`src/data/eda.py::compute_candidate_boundaries` (docs/eda.md Sec.8):
quantiles of the `TransactionDT` *range* (linear interpolation between
dt_min and dt_max), not quantiles of row count -- Phase 1 measured that
transaction volume is not uniform over time (docs/eda.md Sec.8: the first
60% of the time range holds 64.5% of rows), so a row-count quantile would
not actually partition by time.

Boundaries are DT *values*, and partition membership is a strict threshold
comparison (`dt <= boundary` vs `dt > boundary`), not a row-index slice.
This is what makes duplicate-DT ties safe by construction: docs/eda.md
Sec.2 measured 17,191 adjacent duplicate TransactionDT values, and every
row sharing an exact DT value compares identically against any boundary,
so a tied block can never be split across two partitions. See
`tests/test_split.py::test_duplicate_dt_never_straddles_boundary` for a
regression test built specifically to catch a future regression of this
property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class SplitError(Exception):
    """Raised when a split configuration or resulting partition is invalid."""


@dataclass(frozen=True)
class SplitBoundaries:
    dt_min: int
    dt_max: int
    train_val_boundary: int
    val_test_boundary: int
    # Expanding-window CV boundaries, strictly within (dt_min, val_test_boundary],
    # sorted ascending. Fold k uses train = dt <= cv_boundaries[k], eval =
    # (cv_boundaries[k], cv_boundaries[k+1]].
    cv_boundaries: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "train_val_boundary": self.train_val_boundary,
            "val_test_boundary": self.val_test_boundary,
            "cv_boundaries": list(self.cv_boundaries),
        }


def compute_split_boundaries(dt: pd.Series, config: dict[str, Any]) -> SplitBoundaries:
    """Compute DT-range-quantile boundaries for train/val/test and CV folds.

    `config` is the `split:` block of a Phase 2 config, e.g.::

        split:
          train_val_quantile: 0.6
          val_test_quantile: 0.8
          cv_n_folds: 4
    """
    dt_min, dt_max = int(dt.min()), int(dt.max())
    span = dt_max - dt_min
    if span <= 0:
        raise SplitError(f"TransactionDT has zero or negative span (min={dt_min}, max={dt_max})")

    def boundary_at(q: float) -> int:
        return int(round(dt_min + q * span))

    train_val_q = config["train_val_quantile"]
    val_test_q = config["val_test_quantile"]
    if not (0.0 < train_val_q < val_test_q < 1.0):
        raise SplitError(
            f"Require 0 < train_val_quantile < val_test_quantile < 1; got "
            f"train_val_quantile={train_val_q}, val_test_quantile={val_test_q}"
        )

    cv_n_folds = config.get("cv_n_folds", 0)
    if cv_n_folds < 0:
        raise SplitError(f"cv_n_folds must be >= 0, got {cv_n_folds}")
    # cv_n_folds folds need cv_n_folds + 1 boundary points spanning
    # (0, val_test_quantile]; fold k = [boundaries[k], boundaries[k+1]).
    cv_boundaries = tuple(
        boundary_at(val_test_q * i / (cv_n_folds + 1)) for i in range(1, cv_n_folds + 2)
    )

    return SplitBoundaries(
        dt_min=dt_min,
        dt_max=dt_max,
        train_val_boundary=boundary_at(train_val_q),
        val_test_boundary=boundary_at(val_test_q),
        cv_boundaries=cv_boundaries,
    )


def assign_primary_split(dt: pd.Series, boundaries: SplitBoundaries) -> pd.Series:
    """Assign each row to 'train' / 'val' / 'test' by TransactionDT threshold."""
    labels = pd.Series("train", index=dt.index, dtype="object")
    labels[(dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)] = "val"
    labels[dt > boundaries.val_test_boundary] = "test"
    return labels


def assign_cv_eval_fold(dt: pd.Series, boundaries: SplitBoundaries) -> pd.Series:
    """For each row, which CV fold's *eval* window it falls in (nullable Int64).

    A row in the very first bucket (dt <= cv_boundaries[0]) is never an eval
    row (only ever usable as training history for a later fold) and a row
    past `val_test_boundary` (primary test) is excluded from CV entirely --
    both get a null fold id. This column is a compact persisted summary;
    the actual train/eval boolean masks used for fitting come from
    `iter_expanding_window_folds`, not from reconstructing them off this
    column.
    """
    fold = pd.Series(pd.array([pd.NA] * len(dt), dtype="Int64"), index=dt.index)
    edges = boundaries.cv_boundaries
    for k in range(len(edges) - 1):
        mask = (dt > edges[k]) & (dt <= edges[k + 1])
        fold[mask] = k
    return fold


def iter_expanding_window_folds(
    dt: pd.Series, boundaries: SplitBoundaries
) -> list[tuple[int, pd.Series, pd.Series]]:
    """Yield (fold_index, train_mask, eval_mask) for each expanding-window CV fold.

    Fold k: train = {dt <= cv_boundaries[k]}, eval = {cv_boundaries[k] < dt
    <= cv_boundaries[k+1]}. Both masks are boolean Series aligned to `dt`'s
    index. Every fold's train and eval windows lie entirely within (dt_min,
    val_test_boundary] -- the primary test partition is never reachable
    from here (enforced/tested in `src/data/leakage.py`).
    """
    edges = boundaries.cv_boundaries
    folds = []
    for k in range(len(edges) - 1):
        train_mask = dt <= edges[k]
        eval_mask = (dt > edges[k]) & (dt <= edges[k + 1])
        folds.append((k, train_mask, eval_mask))
    return folds


def build_split_assignments(
    transaction_id: pd.Series, dt: pd.Series, boundaries: SplitBoundaries
) -> pd.DataFrame:
    """The persisted split artifact: one row per transaction, split label + CV fold.

    This is the "split indices... saved as a versioned artifact" deliverable
    plan.md Phase 2 requires -- a literal per-row record, not just the
    boundary values, so the exact partition membership is reproducible even
    if the boundary-computation code changes later.
    """
    return pd.DataFrame(
        {
            "TransactionID": transaction_id.to_numpy(),
            "TransactionDT": dt.to_numpy(),
            "split": assign_primary_split(dt, boundaries).to_numpy(),
            "cv_eval_fold": assign_cv_eval_fold(dt, boundaries).to_numpy(),
        }
    )
