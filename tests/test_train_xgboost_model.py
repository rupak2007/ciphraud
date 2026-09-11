import numpy as np
import pandas as pd
import pytest

from src.data.split import compute_split_boundaries
from src.train.xgboost_model import select_hyperparameters_via_cv, train_xgboost

SPLIT_CONFIG = {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 2}


@pytest.fixture
def synthetic_data():
    """Mirrors src/train/data.py's Phase2Features split: train-only data
    (used for the final refit) is kept genuinely distinct from train+val
    (used only for CV selection) -- this distinction is what a real bug
    (docs/baselines.md Sec.6/Sec.8) hid until it hit the real dataset, so
    this fixture is deliberately built to be able to catch a regression of
    it, not just to be minimally sufficient."""
    rng = np.random.RandomState(0)
    n = 800
    dt = pd.Series(np.arange(n) * 100, name="TransactionDT")
    signal = rng.rand(n)
    y = pd.Series((signal > 0.9).astype(int))  # ~10% positive rate
    X = pd.DataFrame(
        {
            "signal_feature": signal.astype("float32"),
            "noise_feature": rng.rand(n).astype("float32"),
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


def test_select_hyperparameters_via_cv_covers_full_grid(synthetic_data):
    d = synthetic_data
    results = select_hyperparameters_via_cv(
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[2, 3], learning_rate_grid=[0.1, 0.2], seed=42,
    )
    assert len(results) == 4
    combos = {(r["max_depth"], r["learning_rate"]) for r in results}
    assert combos == {(2, 0.1), (2, 0.2), (3, 0.1), (3, 0.2)}


def test_select_hyperparameters_via_cv_scores_are_valid(synthetic_data):
    d = synthetic_data
    results = select_hyperparameters_via_cv(
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    for score in results[0]["fold_pr_auc"]:
        assert 0.0 <= score <= 1.0


def test_train_xgboost_selects_best_combo_by_mean_cv_pr_auc(synthetic_data):
    d = synthetic_data
    result = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[2, 4], learning_rate_grid=[0.1], seed=42,
    )
    best_from_cv = max(result.cv_results, key=lambda r: r["mean_pr_auc"])
    assert result.selected_max_depth == best_from_cv["max_depth"]
    assert result.selected_learning_rate == best_from_cv["learning_rate"]


def test_train_xgboost_final_model_uses_early_stopping_n_estimators(synthetic_data):
    """`selected_n_estimators` (best_iteration + 1) is the count of trees
    actually used at prediction time -- NOT the same as
    `num_boosted_rounds()`, which also includes the `early_stopping_rounds`
    patience-window trees XGBoost trains past the best iteration but never
    uses for prediction (verified empirically; see docs/baselines.md
    Sec.2's early-stopping note)."""
    d = synthetic_data
    result = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    assert result.selected_n_estimators >= 1
    # The full patience window is trained and retained in the booster...
    assert result.model.get_booster().num_boosted_rounds() >= result.selected_n_estimators
    # ...but predict_proba() truncates to best_iteration automatically.
    full_range_probs = result.model.predict_proba(
        d["X_val"].to_numpy(), iteration_range=(0, result.model.get_booster().num_boosted_rounds())
    )[:, 1]
    default_probs = result.model.predict_proba(d["X_val"].to_numpy())[:, 1]
    truncated_probs = result.model.predict_proba(
        d["X_val"].to_numpy(), iteration_range=(0, result.selected_n_estimators)
    )[:, 1]
    np.testing.assert_allclose(default_probs, truncated_probs)
    if result.model.get_booster().num_boosted_rounds() > result.selected_n_estimators:
        assert not np.allclose(default_probs, full_range_probs)


def test_train_xgboost_learns_the_real_signal(synthetic_data):
    d = synthetic_data
    result = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    from sklearn.metrics import average_precision_score

    probs = result.model.predict_proba(d["X_val"].to_numpy())[:, 1]
    pr_auc = average_precision_score(d["y_val"].to_numpy(), probs)
    assert pr_auc > 0.5


def test_train_xgboost_save_and_reload_preserves_early_stopping_predictions(synthetic_data, tmp_path):
    """Regression test for the save/load behavior verified empirically
    during Phase 3 development: XGBoost's native save_model()/load_model()
    (used by src/train/pipeline.py, not joblib/pickle) preserves
    best_iteration, so a reloaded model's predict_proba() matches the
    original's without the caller needing to pass iteration_range
    explicitly."""
    d = synthetic_data
    result = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    save_path = tmp_path / "model.json"
    result.model.save_model(str(save_path))

    import xgboost as xgb

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(str(save_path))

    original_probs = result.model.predict_proba(d["X_val"].to_numpy())[:, 1]
    reloaded_probs = reloaded.predict_proba(d["X_val"].to_numpy())[:, 1]
    np.testing.assert_allclose(original_probs, reloaded_probs)


def test_train_xgboost_is_deterministic_given_seed(synthetic_data):
    d = synthetic_data
    result_a = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    result_b = train_xgboost(
        d["X_train"], d["y_train"], d["X_val"], d["y_val"],
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        max_depth_grid=[3], learning_rate_grid=[0.1], seed=42,
    )
    probs_a = result_a.model.predict_proba(d["X_val"].to_numpy())[:, 1]
    probs_b = result_b.model.predict_proba(d["X_val"].to_numpy())[:, 1]
    np.testing.assert_allclose(probs_a, probs_b)
