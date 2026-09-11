"""Evaluate a feature subset with either fixed or re-tuned hyperparameters.

`docs/plan.md` Phase 4: "re-evaluate LR/XGBoost on each tier to quantify
the plaintext accuracy cost of feature reduction." Two distinct questions
need two distinct evaluation modes, so this module offers both rather than
conflating them:

- `evaluate_subset_fixed`: holds Phase 3's already-selected hyperparameters
  fixed and only changes the feature columns -- answers "how much does
  feature count alone cost", isolated from any re-tuning effect. This is
  the fast path used for the dense accuracy-vs-feature-count curve (many
  points; re-running full CV hyperparameter search at every point would
  be prohibitively slow and would also muddy the "feature count alone"
  question with a re-tuning effect).
- `evaluate_tier_retuned`: reuses `train_logistic_regression`/
  `train_xgboost` UNCHANGED on column-subsetted DataFrames -- the official
  tiers (top-20/50/100) each get their own honestly re-tuned
  hyperparameters via Phase 2's expanding-window CV, exactly like Phase 3
  did for the full feature set. This is what actually gets saved as the
  per-tier plaintext reference model for Phase 5+.

Neither function ever touches `X_test`/`y_test` -- both take exactly the
partitions their Phase 3 counterparts take, and the caller
(`src/features/pipeline.py`) is responsible for never passing test data in,
the same discipline `src/train/pipeline.py` follows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.data import split as data_split
from src.train import metrics as train_metrics
from src.train.imbalance import compute_scale_pos_weight
from src.train.logistic_regression import fit_logistic_regression, train_logistic_regression
from src.train.xgboost_model import build_xgb_classifier, train_xgboost


def evaluate_subset_fixed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    columns: list[str],
    lr_c: float,
    lr_max_iter: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Refit both models on `X[columns]` with FIXED (Phase 3) hyperparameters.

    Only `threshold_independent_metrics` (PR-AUC, ROC-AUC) is computed --
    the dense curve this feeds is about ranking feature-count points
    against each other, not about a full per-point threshold analysis.
    """
    X_train_subset = X_train[columns].to_numpy()
    X_val_subset = X_val[columns].to_numpy()
    y_train_arr, y_val_arr = y_train.to_numpy(), y_val.to_numpy()

    lr_model = fit_logistic_regression(X_train_subset, y_train_arr, lr_c, seed, lr_max_iter)
    lr_prob = lr_model.predict_proba(X_val_subset)[:, 1]

    xgb_model = build_xgb_classifier(
        xgb_max_depth, xgb_learning_rate, compute_scale_pos_weight(y_train_arr), seed
    )
    xgb_model.fit(X_train_subset, y_train_arr, eval_set=[(X_val_subset, y_val_arr)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_val_subset)[:, 1]

    return {
        "n_features": len(columns),
        "logistic_regression": train_metrics.threshold_independent_metrics(y_val_arr, lr_prob),
        "xgboost": train_metrics.threshold_independent_metrics(y_val_arr, xgb_prob),
    }


@dataclass
class TierEvaluationResult:
    n_features: int
    lr_model: Pipeline
    xgb_model: xgb.XGBClassifier
    metrics: dict[str, Any]


def evaluate_tier_retuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    cv_dt: pd.Series,
    boundaries: data_split.SplitBoundaries,
    columns: list[str],
    lr_c_grid: list[float],
    lr_max_iter: int,
    xgb_max_depth_grid: list[int],
    xgb_learning_rate_grid: list[float],
    seed: int,
) -> TierEvaluationResult:
    """Per-tier CV re-tuning: the same leakage-safe expanding-window CV
    selection and train-only final refit Phase 3 uses for the full feature
    set, applied to `X[columns]` -- an honestly re-tuned per-tier model,
    not `evaluate_subset_fixed`'s fixed-hyperparameter shortcut."""
    lr_result = train_logistic_regression(
        X_train[columns],
        y_train,
        X_cv[columns],
        y_cv,
        cv_dt,
        boundaries,
        c_grid=lr_c_grid,
        seed=seed,
        max_iter=lr_max_iter,
    )
    xgb_result = train_xgboost(
        X_train[columns],
        y_train,
        X_val[columns],
        y_val,
        X_cv[columns],
        y_cv,
        cv_dt,
        boundaries,
        max_depth_grid=xgb_max_depth_grid,
        learning_rate_grid=xgb_learning_rate_grid,
        seed=seed,
    )

    y_val_arr = y_val.to_numpy()
    lr_prob = lr_result.model.predict_proba(X_val[columns].to_numpy())[:, 1]
    xgb_prob = xgb_result.model.predict_proba(X_val[columns].to_numpy())[:, 1]

    metrics = {
        "n_features": len(columns),
        "logistic_regression": {
            "selected_C": lr_result.selected_c,
            "n_iter": lr_result.n_iter,
            "converged": lr_result.converged,
            "val_evaluation": train_metrics.full_evaluation(y_val_arr, lr_prob),
        },
        "xgboost": {
            "selected_max_depth": xgb_result.selected_max_depth,
            "selected_learning_rate": xgb_result.selected_learning_rate,
            "selected_n_estimators": xgb_result.selected_n_estimators,
            "val_evaluation": train_metrics.full_evaluation(y_val_arr, xgb_prob),
        },
    }
    return TierEvaluationResult(
        n_features=len(columns), lr_model=lr_result.model, xgb_model=xgb_result.model, metrics=metrics
    )


def seed_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    columns: list[str],
    lr_c: float,
    lr_max_iter: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    seeds: list[int],
) -> dict[str, Any]:
    """Val PR-AUC across `seeds`, at FIXED hyperparameters, for a tier.

    Mirrors `tests/test_train_pipeline_real_data.py`'s Phase 3
    seed-stability check, applied per tier: LR (deterministic given fixed
    data) should show ~zero variance; XGBoost (subsample=0.8,
    colsample_bytree=0.8) should show real but bounded variance.
    """
    from sklearn.metrics import average_precision_score

    X_train_subset = X_train[columns].to_numpy()
    X_val_subset = X_val[columns].to_numpy()
    y_train_arr, y_val_arr = y_train.to_numpy(), y_val.to_numpy()

    lr_scores, xgb_scores = [], []
    for seed in seeds:
        lr_model = fit_logistic_regression(X_train_subset, y_train_arr, lr_c, seed, lr_max_iter)
        lr_scores.append(
            float(average_precision_score(y_val_arr, lr_model.predict_proba(X_val_subset)[:, 1]))
        )

        xgb_model = build_xgb_classifier(
            xgb_max_depth, xgb_learning_rate, compute_scale_pos_weight(y_train_arr), seed
        )
        xgb_model.fit(X_train_subset, y_train_arr, eval_set=[(X_val_subset, y_val_arr)], verbose=False)
        xgb_scores.append(
            float(average_precision_score(y_val_arr, xgb_model.predict_proba(X_val_subset)[:, 1]))
        )

    return {
        "seeds": list(seeds),
        "logistic_regression": {"scores": lr_scores, "mean": float(np.mean(lr_scores)), "std": float(np.std(lr_scores))},
        "xgboost": {"scores": xgb_scores, "mean": float(np.mean(xgb_scores)), "std": float(np.std(xgb_scores))},
    }
