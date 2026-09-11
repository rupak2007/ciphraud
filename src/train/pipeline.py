"""Phase 3 baseline-model training + evaluation pipeline.

Run as:
    python -m src.train.pipeline --config configs/phase3/baselines.yaml

Orchestrates, in order:
  1. Load model-ready data from Phase 2 (src/train/data.py) -- the sole
     source of features; nothing here re-derives preprocessing.
  2. Train Logistic Regression (src/train/logistic_regression.py):
     hyperparameter (C) selected via Phase 2's expanding-window CV folds,
     final model refit on the full train partition.
  3. Train XGBoost (src/train/xgboost_model.py): hyperparameters
     (max_depth, learning_rate) selected the same way; n_estimators
     decided by early stopping against X_val for the final refit.
  4. Evaluate both models on VAL ONLY (src/train/metrics.py) -- PR-AUC,
     ROC-AUC, F1-selected and F2-selected thresholds, a fixed-threshold
     sweep, and confusion matrices.
  5. Error analysis (false positive / false negative characterization)
     for whichever model has the higher val PR-AUC.
  6. Save both fitted models, all metrics/CV results, and provenance.

**The test partition (`X_test`/`y_test`) is loaded (Phase 2 always
returns it) but is never passed to any training, selection, or evaluation
call in this file.** This is a deliberate, load-bearing property, not an
oversight -- see the `TEST PARTITION` comment below and
`tests/test_train_pipeline_integration.py::test_test_partition_never_touched`,
which asserts it by patching X_test/y_test into sentinel objects that
raise if any of their methods are ever called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.train import metrics as train_metrics
from src.train.data import load_phase2_features
from src.train.logistic_regression import train_logistic_regression
from src.train.xgboost_model import train_xgboost
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "PyYAML"]


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]

    logger.info("Loading Phase 2 features (sole source of model-ready data)")
    features = load_phase2_features(config["phase2_config"])
    # TEST PARTITION: intentionally not bound to a local variable used
    # anywhere below. features.X_test/features.y_test exist on the
    # returned object (Phase 2 always produces them) but this function
    # body never references them -- the test partition stays untouched
    # until the project's designated final evaluation (not this phase).

    logger.info(
        "Training Logistic Regression",
        extra={"extra_fields": {"c_grid": config["models"]["logistic_regression"]["c_grid"]}},
    )
    lr_result = train_logistic_regression(
        features.X_train,
        features.y_train,
        features.X_train_plus_val,
        features.y_train_plus_val,
        features.train_plus_val_dt,
        features.boundaries,
        c_grid=config["models"]["logistic_regression"]["c_grid"],
        seed=seed,
    )

    logger.info(
        "Training XGBoost",
        extra={
            "extra_fields": {
                "max_depth_grid": config["models"]["xgboost"]["max_depth_grid"],
                "learning_rate_grid": config["models"]["xgboost"]["learning_rate_grid"],
            }
        },
    )
    xgb_result = train_xgboost(
        features.X_train,
        features.y_train,
        features.X_val,
        features.y_val,
        features.X_train_plus_val,
        features.y_train_plus_val,
        features.train_plus_val_dt,
        features.boundaries,
        max_depth_grid=config["models"]["xgboost"]["max_depth_grid"],
        learning_rate_grid=config["models"]["xgboost"]["learning_rate_grid"],
        seed=seed,
    )

    logger.info("Evaluating both models on VAL (test partition untouched)")
    y_val_arr = features.y_val.to_numpy()
    lr_val_prob = lr_result.model.predict_proba(features.X_val.to_numpy())[:, 1]
    xgb_val_prob = xgb_result.model.predict_proba(features.X_val.to_numpy())[:, 1]

    lr_eval = train_metrics.full_evaluation(y_val_arr, lr_val_prob)
    xgb_eval = train_metrics.full_evaluation(y_val_arr, xgb_val_prob)

    strongest = "xgboost" if xgb_eval["pr_auc"] >= lr_eval["pr_auc"] else "logistic_regression"
    strongest_prob = xgb_val_prob if strongest == "xgboost" else lr_val_prob
    strongest_threshold = (xgb_eval if strongest == "xgboost" else lr_eval)[
        f"{config['error_analysis']['threshold_metric']}_selected"
    ]["selected_threshold"]
    logger.info(
        "Running error analysis for strongest model",
        extra={"extra_fields": {"strongest_model": strongest, "threshold": strongest_threshold}},
    )
    error_report = train_metrics.error_analysis(
        features.X_val.index, y_val_arr, strongest_prob, strongest_threshold
    )

    logger.info("Saving model artifacts")
    joblib.dump(lr_result.model, models_dir / "logistic_regression.joblib")
    xgb_result.model.save_model(str(models_dir / "xgboost.json"))

    logger.info("Writing results")
    results = {
        "logistic_regression": {
            "selected_C": lr_result.selected_c,
            "cv_results": lr_result.cv_results,
            "val_evaluation": lr_eval,
        },
        "xgboost": {
            "selected_max_depth": xgb_result.selected_max_depth,
            "selected_learning_rate": xgb_result.selected_learning_rate,
            "selected_n_estimators": xgb_result.selected_n_estimators,
            "cv_results": xgb_result.cv_results,
            "val_evaluation": xgb_eval,
        },
        "strongest_model": strongest,
        "error_analysis": error_report,
        "split_sizes": {
            "train": len(features.X_train),
            "val": len(features.X_val),
            "test": len(features.X_test),
        },
        "test_partition_touched": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    provenance = data_provenance.build_provenance(
        config=config,
        config_path=Path(config_path),
        seed=seed,
        raw_file_digests={},  # inherited from Phase 2's own provenance; not re-verified here
        library_names=LIBRARY_NAMES,
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    logger.info(
        "Phase 3 baseline training complete",
        extra={
            "extra_fields": {
                "lr_val_pr_auc": lr_eval["pr_auc"],
                "xgb_val_pr_auc": xgb_eval["pr_auc"],
                "strongest_model": strongest,
            }
        },
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase3/baselines.yaml", help="Path to the Phase 3 baselines config YAML"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
