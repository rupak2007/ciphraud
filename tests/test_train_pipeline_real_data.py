"""Phase 3 pipeline integration test against the REAL IEEE-CIS dataset.

Skipped when Phase 2's model-ready data isn't available (mirroring every
other *_real_data.py test in this repo). When present, this proves the
full Phase 3 training/evaluation pipeline runs end-to-end on real data,
and includes the seed-stability check docs/plan.md Phase 3 requires
("metrics stable within a documented variance band across seeds").

The pipeline (and a small seed-stability re-fit) runs ONCE per test
session (module-scoped fixtures) -- real data at this scale takes real
time, and each test asserts a different property of that single run
rather than re-running it.
"""

import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.train.data import load_phase2_features
from src.train.pipeline import run

_PHASE3_CONFIG_PATH = "configs/phase3/baselines.yaml"


def _real_data_available() -> bool:
    config = load_config(_PHASE3_CONFIG_PATH)
    phase2_config = load_config(config["phase2_config"])
    raw_dir = PROJECT_ROOT / phase2_config["data"]["raw_dir"]
    return (raw_dir / phase2_config["data"]["transaction_file"]).exists() and (
        raw_dir / phase2_config["data"]["identity_file"]
    ).exists()


pytestmark = pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Real IEEE-CIS data not found under data/raw/. See "
        "docs/data_acquisition.md for the manual download procedure."
    ),
)


@pytest.fixture(scope="module")
def pipeline_result():
    return run(_PHASE3_CONFIG_PATH)


def test_pipeline_runs_end_to_end_on_real_data(pipeline_result):
    assert "logistic_regression" in pipeline_result
    assert "xgboost" in pipeline_result


def test_pipeline_real_data_pr_auc_beats_baseline(pipeline_result):
    """PR-AUC for a genuinely useful model must clear the positive-rate
    baseline (~3.5%, docs/eda.md Sec.1) by a wide margin -- this is the
    project's own stated risk ("class imbalance handling insufficient,
    producing misleadingly high accuracy but poor PR-AUC", docs/plan.md
    Phase 3), checked directly rather than assumed."""
    lr_pr_auc = pipeline_result["logistic_regression"]["val_evaluation"]["pr_auc"]
    xgb_pr_auc = pipeline_result["xgboost"]["val_evaluation"]["pr_auc"]
    assert lr_pr_auc > 0.15, f"LR PR-AUC {lr_pr_auc} too close to the ~3.5% baseline"
    assert xgb_pr_auc > 0.15, f"XGBoost PR-AUC {xgb_pr_auc} too close to the ~3.5% baseline"


def test_pipeline_real_data_metrics_are_not_accuracy_only(pipeline_result):
    """CLAUDE.md Sec.6: accuracy must never be the headline metric."""
    for model_key in ("logistic_regression", "xgboost"):
        val_eval = pipeline_result[model_key]["val_evaluation"]
        assert "pr_auc" in val_eval
        assert "roc_auc" in val_eval
        assert "accuracy" not in val_eval


def test_pipeline_real_data_logistic_regression_converges(pipeline_result):
    """Regression guard for the real bug documented in docs/baselines.md
    Sec.7.3: LR trained on the unscaled real feature matrix (TransactionAmt
    up to ~$31,937 alongside [0,1] frequency encodings) failed to converge
    within lbfgs's iteration budget. The fix adds a train-only-fit
    StandardScaler; this asserts the fix actually holds on real data,
    not just on synthetic fixtures."""
    lr_result = pipeline_result["logistic_regression"]
    assert lr_result["converged"] is True, f"LR did not converge: n_iter={lr_result['n_iter']}"


def test_pipeline_real_data_test_partition_not_in_results(pipeline_result):
    assert pipeline_result["test_partition_touched"] is False
    import json

    output_dir = PROJECT_ROOT / load_config(_PHASE3_CONFIG_PATH)["output"]["dir"]
    raw_text = (output_dir / "metrics.json").read_text()
    # The strongest structural guarantee this text-level check can offer:
    # no test-set prediction/probability array was ever serialized here.
    parsed = json.loads(raw_text)
    assert "test_evaluation" not in parsed.get("logistic_regression", {})
    assert "test_evaluation" not in parsed.get("xgboost", {})


@pytest.fixture(scope="module")
def seed_stability_results(pipeline_result):
    """Re-fit the FINAL chosen configuration (hyperparameters already
    selected once via CV inside `pipeline_result`, reused here rather than
    re-running the grid search) across 3 seeds and report val PR-AUC
    variance. docs/plan.md Phase 3 Tests: 'seed-stability check (metrics
    stable within a documented variance band across seeds)'."""
    features = load_phase2_features("configs/phase2/pipeline.yaml")
    from sklearn.metrics import average_precision_score

    from src.train.imbalance import compute_scale_pos_weight
    from src.train.logistic_regression import fit_logistic_regression as _fit_lr
    from src.train.xgboost_model import build_xgb_classifier as _build_xgb

    best_c = pipeline_result["logistic_regression"]["selected_C"]
    best_depth = pipeline_result["xgboost"]["selected_max_depth"]
    best_lr = pipeline_result["xgboost"]["selected_learning_rate"]
    max_iter = load_config(_PHASE3_CONFIG_PATH)["models"]["logistic_regression"]["max_iter"]

    seeds = [42, 43, 44]
    lr_scores, xgb_scores = [], []
    for seed in seeds:
        lr_model = _fit_lr(features.X_train.to_numpy(), features.y_train.to_numpy(), best_c, seed, max_iter)
        lr_scores.append(
            average_precision_score(features.y_val, lr_model.predict_proba(features.X_val.to_numpy())[:, 1])
        )

        xgb_model = _build_xgb(
            best_depth, best_lr, compute_scale_pos_weight(features.y_train.to_numpy()), seed
        )
        xgb_model.fit(
            features.X_train.to_numpy(), features.y_train.to_numpy(),
            eval_set=[(features.X_val.to_numpy(), features.y_val.to_numpy())], verbose=False,
        )
        xgb_scores.append(
            average_precision_score(features.y_val, xgb_model.predict_proba(features.X_val.to_numpy())[:, 1])
        )

    return {"lr_scores": lr_scores, "xgb_scores": xgb_scores}


def test_seed_stability_logistic_regression_is_exactly_deterministic(seed_stability_results):
    """lbfgs LogisticRegression has no meaningful seed-dependent randomness
    at this scale -- report that finding directly rather than assume it;
    variance should be at or extremely near zero."""
    scores = seed_stability_results["lr_scores"]
    assert np.std(scores) < 0.001, f"Unexpected LR seed variance: {scores}"


def test_seed_stability_xgboost_within_documented_band(seed_stability_results):
    """subsample=0.8/colsample_bytree=0.8 (src/train/xgboost_model.py) make
    seed meaningfully affect training; the documented acceptable band is
    std(PR-AUC) < 0.02 across seeds -- a real check, not a rubber stamp."""
    scores = seed_stability_results["xgb_scores"]
    assert np.std(scores) < 0.02, f"XGBoost PR-AUC too unstable across seeds: {scores}"
