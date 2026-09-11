"""Adapts Phase 2's expanding-window CV folds into positional index arrays.

`src/data/split.py::iter_expanding_window_folds` yields boolean masks
aligned to a `TransactionDT` Series' index. Model-selection code here
needs 0-based *positional* index arrays into `X_train` (matching
`scikit-learn`'s `cv=` iterable-of-(train_idx, test_idx) convention and
XGBoost's `.iloc`-based row selection) -- this module is the one place
that conversion happens, so `logistic_regression.py` and
`xgboost_model.py` share identical fold semantics rather than each
re-deriving it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import split as data_split


def build_positional_cv_folds(
    train_dt: pd.Series, boundaries: data_split.SplitBoundaries
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Returns [(fold_index, train_positions, eval_positions), ...].

    `train_dt` must be indexed/ordered exactly as the feature matrix these
    positions will be applied to (`src/train/data.py::Phase2Features.train_dt`
    is built for exactly this). Positions are 0-based row positions, not
    TransactionIDs or pandas index labels.
    """
    folds = data_split.iter_expanding_window_folds(train_dt, boundaries)
    positional_folds = []
    for fold_index, train_mask, eval_mask in folds:
        train_positions = np.flatnonzero(train_mask.to_numpy())
        eval_positions = np.flatnonzero(eval_mask.to_numpy())
        positional_folds.append((fold_index, train_positions, eval_positions))
    return positional_folds
