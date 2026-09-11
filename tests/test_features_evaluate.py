import numpy as np
import pandas as pd
import pytest

from src.data.split import compute_split_boundaries
from src.features.evaluate import evaluate_subset_fixed, evaluate_tier_retuned, seed_stability

SPLIT_CONFIG = {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 2}
MAX_ITER = 200


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(0)
    n = 800
    dt = pd.Series(np.arange(n) * 100, name="TransactionDT")
    signal = rng.rand(n)
    y = pd.Series((signal > 0.9).astype(int))
    X = pd.DataFrame(
        {
            "signal_feature": signal.astype("float32"),
            "noise_feature_1": rng.rand(n).astype("float32"),
            "noise_feature_2": rng.rand(n).astype("float32"),
        }
    )
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)
    train_mask = (dt <= boundaries.train_val_boundary).to_numpy()
    val_mask = ((dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)).to_numpy()
    train_plus_val_mask = train_mask | val_mask

    return {
        "X_train": X.loc[train_mask],
        "y_train": y.loc[train_mask],
        "X_val": X.loc[val_mask],
        "y_val": y.loc[val_mask],
        "X_cv": X.loc[train_plus_val_mask],
        "y_cv": y.loc[train_plus_val_mask],
        "cv_dt": dt.loc[train_plus_val_mask],
        "boundaries": boundaries,
    }


def test_evaluate_subset_fixed_reports_n_features(synthetic_data):
    d = synthetic_data
    result = evaluate_subset_fixed(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"], ["signal_feature"],
        lr_c=1.0, lr_max_iter=MAX_ITER, xgb_max_depth=3, xgb_learning_rate=0.1, seed=42,
    )
    assert result["n_features"] == 1
    assert 0.0 <= result["logistic_regression"]["pr_auc"] <= 1.0
    assert 0.0 <= result["xgboost"]["pr_auc"] <= 1.0


def test_evaluate_subset_fixed_multiple_columns_does_not_crash(synthetic_data):
    d = synthetic_data
    columns = ["signal_feature", "noise_feature_1", "noise_feature_2"]
    result = evaluate_subset_fixed(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"], columns,
        lr_c=1.0, lr_max_iter=MAX_ITER, xgb_max_depth=3, xgb_learning_rate=0.1, seed=42,
    )
    assert result["n_features"] == 3


def test_evaluate_tier_retuned_final_models_see_only_tier_columns(synthetic_data):
    d = synthetic_data
    columns = ["signal_feature", "noise_feature_1"]
    result = evaluate_tier_retuned(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], columns,
        lr_c_grid=[0.1, 1.0], lr_max_iter=MAX_ITER,
        xgb_max_depth_grid=[3], xgb_learning_rate_grid=[0.1], seed=42,
    )
    assert result.n_features == 2
    assert result.lr_model.n_features_in_ == 2
    assert result.xgb_model.n_features_in_ == 2
    assert "val_evaluation" in result.metrics["logistic_regression"]
    assert "val_evaluation" in result.metrics["xgboost"]


def test_evaluate_tier_retuned_learns_the_real_signal(synthetic_data):
    d = synthetic_data
    result = evaluate_tier_retuned(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], ["signal_feature"],
        lr_c_grid=[1.0], lr_max_iter=MAX_ITER,
        xgb_max_depth_grid=[3], xgb_learning_rate_grid=[0.1], seed=42,
    )
    assert result.metrics["xgboost"]["val_evaluation"]["pr_auc"] > 0.3


def test_seed_stability_reports_mean_and_std(synthetic_data):
    d = synthetic_data
    columns = ["signal_feature", "noise_feature_1"]
    result = seed_stability(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"], columns,
        lr_c=1.0, lr_max_iter=MAX_ITER, xgb_max_depth=3, xgb_learning_rate=0.1, seeds=[1, 2, 3],
    )
    assert len(result["logistic_regression"]["scores"]) == 3
    assert len(result["xgboost"]["scores"]) == 3
    # LR is deterministic given fixed data (lbfgs, no row/feature subsampling).
    assert result["logistic_regression"]["std"] < 1e-6
