"""Logistic Regression baseline (CLAUDE.md Sec.6 required baseline #1).

Hyperparameter selection (`C`, inverse regularization strength) uses
Phase 2's expanding-window CV folds (`src/train/cv.py`) -- never random
k-fold (CLAUDE.md Sec.6) -- scored by mean PR-AUC across folds. CV
selection is deliberately given `(X_cv, y_cv, cv_dt)` -- Phase 2's CV
folds span the primary train+val region by design (docs/pipeline.md
Sec.1), so this must be `features.X_train_plus_val` etc.
(`src/train/data.py`), NOT `features.X_train` alone: the latter's DT range
stops at `train_val_boundary` and cannot cover the last fold's eval
window, which falls inside the primary "val" partition's own DT range
(this was a real bug, caught by the real-data test producing a
zero-row fold -- see `docs/baselines.md` Sec.6/Sec.8). The final model is
then refit on `(X_train, y_train)` ONLY -- never val -- keeping val
reserved for the model's headline evaluation
(`src/train/pipeline.py`), never touched during hyperparameter selection.

The model itself is `sklearn.linear_model.LogisticRegression` wrapped in an
`sklearn.pipeline.Pipeline` with a `StandardScaler` as its only other step
-- not a bare, unwrapped `LogisticRegression` -- because a real run against
the actual Phase 2 feature matrix (`TransactionAmt` up to ~$31,937 sitting
alongside frequency encodings in [0, 1] and 0/1 indicators) never converged
within `lbfgs`'s default iteration budget (16 `ConvergenceWarning`s on the
committed Phase 3 run; see `docs/baselines.md` Sec.7.3). `Pipeline.fit`
fits `StandardScaler` only on the rows it is given, so it is train-only by
construction in exactly the same way the final `LogisticRegression` step
is -- both in each CV fold (fit on that fold's train window only) and in
the final refit (fit on `X_train` only). This is still FHE-compatible:
per `docs/architecture.md` Sec.6, feature preprocessing (which now
includes this scaler) stays client-side and plaintext, and
`concrete.ml.sklearn.LogisticRegression` (Phase 5+) remains a drop-in
replacement for the pipeline's `lr` step over the scaler's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train.cv import build_positional_cv_folds
from src.train.imbalance import class_weight_dict
from src.data import split as data_split
from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class LogisticRegressionResult:
    model: Pipeline
    selected_c: float
    cv_results: list[dict[str, Any]]
    n_iter: int
    converged: bool


def _n_iter_and_converged(model: Pipeline, max_iter: int) -> tuple[int, bool]:
    """`LogisticRegression.n_iter_` is an array (one entry per class for
    some solvers); `lbfgs` on a binary problem gives a single-element
    array. Take its max so a partial-convergence edge case can't be
    silently missed."""
    n_iter = int(np.max(model.named_steps["lr"].n_iter_))
    return n_iter, n_iter < max_iter


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, c: float, seed: int, max_iter: int) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight_dict(y),
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def select_c_via_cv(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    cv_dt: pd.Series,
    boundaries: data_split.SplitBoundaries,
    c_grid: list[float],
    seed: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    """Mean CV PR-AUC for each candidate `C`, using Phase 2's expanding-window folds.

    `X_cv`/`y_cv`/`cv_dt` must span the primary train+val region (e.g.
    `features.X_train_plus_val` from `src/train/data.py`) -- see module
    docstring for why train-only data breaks the last fold.
    """
    folds = build_positional_cv_folds(cv_dt, boundaries)
    X_arr, y_arr = X_cv.to_numpy(), y_cv.to_numpy()

    results = []
    for c in c_grid:
        fold_scores = []
        fold_n_iter = []
        fold_converged = []
        for fold_index, train_pos, eval_pos in folds:
            model = fit_logistic_regression(X_arr[train_pos], y_arr[train_pos], c, seed, max_iter)
            y_prob = model.predict_proba(X_arr[eval_pos])[:, 1]
            score = float(average_precision_score(y_arr[eval_pos], y_prob))
            fold_scores.append(score)
            n_iter, converged = _n_iter_and_converged(model, max_iter)
            fold_n_iter.append(n_iter)
            fold_converged.append(converged)
            logger.info(
                "LR CV fold scored",
                extra={
                    "extra_fields": {
                        "C": c,
                        "fold_index": fold_index,
                        "pr_auc": score,
                        "n_iter": n_iter,
                        "converged": converged,
                    }
                },
            )
        results.append(
            {
                "C": c,
                "fold_pr_auc": fold_scores,
                "mean_pr_auc": float(np.mean(fold_scores)),
                "std_pr_auc": float(np.std(fold_scores)),
                "fold_n_iter": fold_n_iter,
                "fold_converged": fold_converged,
            }
        )
    return results


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    cv_dt: pd.Series,
    boundaries: data_split.SplitBoundaries,
    c_grid: list[float],
    seed: int,
    max_iter: int,
) -> LogisticRegressionResult:
    """`(X_train, y_train)`: train-only, used for the final refit.
    `(X_cv, y_cv, cv_dt)`: train+val, used only for CV-based `C` selection.
    See module docstring for why these must NOT be the same data."""
    cv_results = select_c_via_cv(X_cv, y_cv, cv_dt, boundaries, c_grid, seed, max_iter)
    best = max(cv_results, key=lambda r: r["mean_pr_auc"])
    selected_c = best["C"]
    logger.info(
        "Selected LR hyperparameter via CV",
        extra={"extra_fields": {"selected_C": selected_c, "mean_cv_pr_auc": best["mean_pr_auc"]}},
    )

    final_model = fit_logistic_regression(X_train.to_numpy(), y_train.to_numpy(), selected_c, seed, max_iter)
    n_iter, converged = _n_iter_and_converged(final_model, max_iter)
    if not converged:
        logger.warning(
            "Final LogisticRegression did not converge within max_iter",
            extra={"extra_fields": {"max_iter": max_iter, "n_iter": n_iter}},
        )
    return LogisticRegressionResult(
        model=final_model, selected_c=selected_c, cv_results=cv_results, n_iter=n_iter, converged=converged
    )
