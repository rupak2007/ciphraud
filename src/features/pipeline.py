"""Phase 4 feature-importance ranking, tier construction, and per-tier evaluation.

Run as:
    python -m src.features.pipeline --config configs/phase4/features.yaml

Orchestrates, in order:
  1. Load model-ready data from Phase 2 (src/train/data.py) -- the sole
     source of features, exactly as Phase 3 does.
  2. Verify Phase 3's committed model/metrics/config are internally
     consistent with the current on-disk state (`_verify_phase3_integrity`)
     -- fails loudly with "run Phase 3 first" rather than silently ranking
     against a stale or hand-edited model.
  3. Rank all 832 features by the Phase 3 XGBoost model's `total_gain`
     (src/features/importance.py).
  4. Measure ranking stability across two additional seeds (Jaccard
     overlap of top-k sets vs. the Phase 3 seed).
  5. Compute a dense accuracy-vs-feature-count curve at fixed (Phase 3)
     hyperparameters -- the full 832-feature point on this curve doubles
     as a re-fit reproduction check against Phase 3's own committed
     metrics.
  6. Build the official tiers (top-20/50/100 by default), re-tune each via
     Phase 2's expanding-window CV exactly like Phase 3 does, measure
     XGBoost seed stability per tier, and save both models per tier.
  7. Write `tiers.json` (the versioned artifact every later FHE phase
     reads through `src/features/tiers.py::load_tiers`), `metrics.json`,
     and `provenance.json`.

**The test partition (`X_test`/`y_test`) is loaded (Phase 2 always returns
it) but is never passed to any ranking, tuning, or evaluation call in this
file** -- the same deliberate, load-bearing property
`src/train/pipeline.py` has, verified the same way:
`tests/test_features_pipeline_integration.py::test_test_partition_never_touched`
patches X_test/y_test into sentinel objects that raise if any of their
methods besides `__len__` are ever called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.features.evaluate import evaluate_subset_fixed, evaluate_tier_retuned, seed_stability
from src.features.importance import topk_jaccard, xgboost_total_gain_ranking
from src.features.tiers import build_tiers, save_tiers, v_block_coverage
from src.logging_setup import get_logger
from src.train.data import load_phase2_features
from src.train.imbalance import compute_scale_pos_weight
from src.train.xgboost_model import build_xgb_classifier

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "PyYAML"]


class Phase3IntegrityError(Exception):
    """Raised when Phase 3's committed artifacts don't match the current on-disk state."""


# Tolerance for _verify_phase3_integrity's reload-the-saved-artifact check
# -- NOT config-driven, unlike run()'s refit-based reproduction check
# below. Reloading a saved model and recomputing predict_proba involves no
# re-fitting, so it should match the committed metric to near-exact
# floating-point precision; a real discrepancy here means the model file,
# metrics.json, or the current feature matrix have genuinely drifted
# apart, which no config value should be able to paper over.
_INTEGRITY_RELOAD_TOLERANCE = 1e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of_names(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def _verify_phase3_integrity(
    phase3_config: dict[str, Any], phase3_config_path: str, features
) -> dict[str, Any]:
    """Fails loudly if Phase 3 hasn't been (re-)run, or if its committed
    model/metrics no longer match the current Phase 2 feature matrix.
    Reloads the ACTUAL SAVED model artifacts (not a re-fit) and recomputes
    their val PR-AUC -- this ties model file <-> committed metrics <->
    current feature order together, distinct from the re-fit reproduction
    check `run()` performs later against the accuracy-vs-feature-count
    curve's full-feature point. Uses `_INTEGRITY_RELOAD_TOLERANCE`, not the
    config's (much looser) `reproduction_check.pr_auc_tolerance` -- a
    reload-and-predict involves no re-fitting, so it should match to
    near-exact precision; see that constant's docstring for why the two
    checks deliberately use different tolerances.
    """
    output_dir = PROJECT_ROOT / phase3_config["output"]["dir"]
    models_dir = PROJECT_ROOT / phase3_config["output"]["models_dir"]

    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    lr_path = models_dir / "logistic_regression.joblib"
    xgb_path = models_dir / "xgboost.json"

    missing = [p for p in (metrics_path, provenance_path, lr_path, xgb_path) if not p.exists()]
    if missing:
        raise Phase3IntegrityError(
            f"Phase 3 artifacts missing: {[str(p) for p in missing]} -- run Phase 3 first "
            f"(python -m src.train.pipeline --config {phase3_config_path})"
        )

    recorded_hash = json.loads(provenance_path.read_text())["config_hash"]
    current_hash = data_provenance.compute_config_hash(phase3_config)
    if recorded_hash != current_hash:
        raise Phase3IntegrityError(
            f"Phase 3 config_hash mismatch (committed={recorded_hash}, current={current_hash}) -- "
            f"{phase3_config_path} changed since the committed run; re-run Phase 3 first"
        )

    committed_metrics = json.loads(metrics_path.read_text())

    lr_model = joblib.load(lr_path)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_path))

    X_val = features.X_val.to_numpy()
    y_val = features.y_val.to_numpy()
    lr_pr_auc = float(average_precision_score(y_val, lr_model.predict_proba(X_val)[:, 1]))
    xgb_pr_auc = float(average_precision_score(y_val, xgb_model.predict_proba(X_val)[:, 1]))

    committed_lr_pr_auc = committed_metrics["logistic_regression"]["val_evaluation"]["pr_auc"]
    committed_xgb_pr_auc = committed_metrics["xgboost"]["val_evaluation"]["pr_auc"]

    if not np.isclose(lr_pr_auc, committed_lr_pr_auc, atol=_INTEGRITY_RELOAD_TOLERANCE, rtol=0):
        raise Phase3IntegrityError(
            f"Reloaded LR model's val PR-AUC ({lr_pr_auc}) does not match committed "
            f"metrics.json ({committed_lr_pr_auc}) within tolerance {_INTEGRITY_RELOAD_TOLERANCE}"
        )
    if not np.isclose(xgb_pr_auc, committed_xgb_pr_auc, atol=_INTEGRITY_RELOAD_TOLERANCE, rtol=0):
        raise Phase3IntegrityError(
            f"Reloaded XGBoost model's val PR-AUC ({xgb_pr_auc}) does not match committed "
            f"metrics.json ({committed_xgb_pr_auc}) within tolerance {_INTEGRITY_RELOAD_TOLERANCE}"
        )

    logger.info(
        "Phase 3 integrity verified", extra={"extra_fields": {"lr_pr_auc": lr_pr_auc, "xgb_pr_auc": xgb_pr_auc}}
    )
    return {"xgb_model": xgb_model, "committed_metrics": committed_metrics}


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    tolerance = config["reproduction_check"]["pr_auc_tolerance"]

    logger.info("Loading Phase 2 features (sole source of model-ready data)")
    features = load_phase2_features(config["phase2_config"])
    feature_names = list(features.X_train.columns)
    # TEST PARTITION: intentionally not bound to a local variable used
    # anywhere below except len() in the final split_sizes report -- see
    # module docstring and src/train/pipeline.py's identical discipline.

    phase3_config = load_config(config["phase3_config"])
    integrity = _verify_phase3_integrity(phase3_config, config["phase3_config"], features)
    xgb_model = integrity["xgb_model"]
    phase3_metrics = integrity["committed_metrics"]
    lr_c = phase3_metrics["logistic_regression"]["selected_C"]
    lr_max_iter = phase3_config["models"]["logistic_regression"]["max_iter"]
    xgb_max_depth = phase3_metrics["xgboost"]["selected_max_depth"]
    xgb_learning_rate = phase3_metrics["xgboost"]["selected_learning_rate"]

    logger.info("Ranking features by XGBoost total_gain")
    ranking = xgboost_total_gain_ranking(xgb_model.get_booster(), feature_names)
    ranking.to_csv(output_dir / "feature_ranking.csv", index=False)

    tier_sizes = config["tiers"]["sizes"]
    stability_ks = sorted(set(tier_sizes) | set(config["curve"]["feature_counts"]))

    logger.info("Measuring ranking stability across seeds", extra={"extra_fields": {"seeds": config["importance"]["ranking_stability_seeds"]}})
    ranking_stability_rows = []
    for stability_seed in config["importance"]["ranking_stability_seeds"]:
        stability_xgb = build_xgb_classifier(
            xgb_max_depth, xgb_learning_rate, compute_scale_pos_weight(features.y_train.to_numpy()), stability_seed
        )
        stability_xgb.fit(
            features.X_train.to_numpy(),
            features.y_train.to_numpy(),
            eval_set=[(features.X_val.to_numpy(), features.y_val.to_numpy())],
            verbose=False,
        )
        stability_ranking = xgboost_total_gain_ranking(stability_xgb.get_booster(), feature_names)
        for k in stability_ks:
            ranking_stability_rows.append(
                {"seed": stability_seed, "k": k, "jaccard_vs_phase3_seed": topk_jaccard(ranking, stability_ranking, k)}
            )
    pd.DataFrame(ranking_stability_rows).to_csv(output_dir / "ranking_stability.csv", index=False)

    logger.info("Computing accuracy-vs-feature-count curve")
    n_features_total = len(feature_names)
    # NOTE: every curve point (including the full-feature-set point, when
    # config["curve"]["feature_counts"] includes n_features_total) selects
    # columns in RANK order, not Phase 2's original column order -- this is
    # intentional and consistent across the whole curve (it's what "top-k by
    # importance" means), but it is NOT the same model as Phase 3's original
    # fit even at k=n_features_total, since column order is not just cosmetic
    # for either model type (see _reproduce_phase3_full_feature_fit below).
    # The reproduction check therefore uses a SEPARATE refit in Phase 2's
    # actual column order, never this curve's own full-feature-set row.
    curve_ks = sorted(set(config["curve"]["feature_counts"]) | {n_features_total})
    curve_rows = []
    for k in curve_ks:
        columns = ranking.sort_values("rank")["feature"].head(k).tolist()
        curve_rows.append(
            evaluate_subset_fixed(
                features.X_train, features.y_train, features.X_val, features.y_val,
                columns, lr_c, lr_max_iter, xgb_max_depth, xgb_learning_rate, seed,
            )
        )
    pd.json_normalize(curve_rows).to_csv(output_dir / "accuracy_vs_feature_count.csv", index=False)

    # Reproduction check: a refit on ALL features in Phase 2's ORIGINAL
    # column order (feature_names), with Phase 3's exact hyperparameters and
    # seed, must reproduce Phase 3's own committed val PR-AUC almost exactly.
    #
    # This is a RE-FIT determinism check, distinct from
    # _verify_phase3_integrity's reload-the-saved-artifact check above.
    # Column order matters here for real, measured reasons, not just as a
    # theoretical concern -- see docs/features.md Sec.6.1 for the controlled
    # experiment (refit in Phase 2 order vs. rank order) that found this:
    # XGBoost's `colsample_bytree=0.8` samples a random subset of column
    # POSITIONS each split, so a different column order changes which named
    # features that subset actually contains; LR's tiny remaining diff is
    # consistent with floating-point summation-order sensitivity in lbfgs's
    # gradient computation. An earlier version of this check reused the
    # curve's rank-ordered full-feature-set row and misattributed the
    # resulting (deterministic, reproducible) diff to XGBoost thread
    # nondeterminism -- an incorrect diagnosis, corrected here.
    reproduction_row = evaluate_subset_fixed(
        features.X_train, features.y_train, features.X_val, features.y_val,
        feature_names, lr_c, lr_max_iter, xgb_max_depth, xgb_learning_rate, seed,
    )
    lr_diff = abs(reproduction_row["logistic_regression"]["pr_auc"] - phase3_metrics["logistic_regression"]["val_evaluation"]["pr_auc"])
    xgb_diff = abs(reproduction_row["xgboost"]["pr_auc"] - phase3_metrics["xgboost"]["val_evaluation"]["pr_auc"])
    if lr_diff > tolerance or xgb_diff > tolerance:
        raise Phase3IntegrityError(
            f"Full-feature-set refit (Phase 2 column order) does not reproduce Phase 3's "
            f"committed val PR-AUC (LR diff={lr_diff}, XGBoost diff={xgb_diff}, tolerance={tolerance})"
        )

    logger.info("Building official tiers", extra={"extra_fields": {"sizes": tier_sizes}})
    tiers = build_tiers(ranking, tier_sizes)
    blocks_csv = PROJECT_ROOT / "results" / "phase1_eda" / "v_null_mask_blocks.csv"
    v_block_report = {name: v_block_coverage(members, blocks_csv) for name, members in tiers.items()}

    logger.info("Evaluating and re-tuning each official tier")
    full_lr_pr_auc = phase3_metrics["logistic_regression"]["val_evaluation"]["pr_auc"]
    full_xgb_pr_auc = phase3_metrics["xgboost"]["val_evaluation"]["pr_auc"]
    tier_results: dict[str, Any] = {}
    for tier_name, columns in tiers.items():
        tier_dir = models_dir / tier_name
        tier_dir.mkdir(parents=True, exist_ok=True)

        retuned = evaluate_tier_retuned(
            features.X_train, features.y_train, features.X_val, features.y_val,
            features.X_train_plus_val, features.y_train_plus_val, features.train_plus_val_dt,
            features.boundaries, columns,
            lr_c_grid=phase3_config["models"]["logistic_regression"]["c_grid"],
            lr_max_iter=lr_max_iter,
            xgb_max_depth_grid=phase3_config["models"]["xgboost"]["max_depth_grid"],
            xgb_learning_rate_grid=phase3_config["models"]["xgboost"]["learning_rate_grid"],
            seed=seed,
        )
        joblib.dump(retuned.lr_model, tier_dir / "logistic_regression.joblib")
        retuned.xgb_model.save_model(str(tier_dir / "xgboost.json"))

        tier_stability = seed_stability(
            features.X_train, features.y_train, features.X_val, features.y_val, columns,
            lr_c=retuned.metrics["logistic_regression"]["selected_C"],
            lr_max_iter=lr_max_iter,
            xgb_max_depth=retuned.metrics["xgboost"]["selected_max_depth"],
            xgb_learning_rate=retuned.metrics["xgboost"]["selected_learning_rate"],
            seeds=config["stability"]["tier_eval_seeds"],
        )

        tier_lr_pr_auc = retuned.metrics["logistic_regression"]["val_evaluation"]["pr_auc"]
        tier_xgb_pr_auc = retuned.metrics["xgboost"]["val_evaluation"]["pr_auc"]
        tier_results[tier_name] = {
            **retuned.metrics,
            "seed_stability": tier_stability,
            "v_block_coverage": v_block_report[tier_name],
            "pr_auc_retention": {
                "logistic_regression": tier_lr_pr_auc / full_lr_pr_auc if full_lr_pr_auc else None,
                "xgboost": tier_xgb_pr_auc / full_xgb_pr_auc if full_xgb_pr_auc else None,
            },
        }
        logger.info(
            "Tier evaluated",
            extra={
                "extra_fields": {
                    "tier": tier_name, "n_features": len(columns),
                    "lr_pr_auc": tier_lr_pr_auc, "xgb_pr_auc": tier_xgb_pr_auc,
                }
            },
        )

    tiers_path = output_dir / "tiers.json"
    save_tiers(
        tiers_path,
        tiers,
        metadata={
            "source_model_sha256": _sha256_file(PROJECT_ROOT / phase3_config["output"]["models_dir"] / "xgboost.json"),
            "phase2_feature_list_sha256": _sha256_of_names(feature_names),
            "seed": seed,
            "importance_method": config["importance"]["method"],
        },
    )

    results = {
        "ranking_head": ranking.sort_values("rank").head(20).to_dict(orient="records"),
        "tier_sizes": tier_sizes,
        "tiers": tier_results,
        "curve_points": len(curve_rows),
        "reproduction_check": {"lr_diff": lr_diff, "xgb_diff": xgb_diff, "tolerance": tolerance},
        "split_sizes": {"train": len(features.X_train), "val": len(features.X_val), "test": len(features.X_test)},
        "test_partition_touched": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    provenance = data_provenance.build_provenance(
        config=config, config_path=Path(config_path), seed=seed, raw_file_digests={}, library_names=LIBRARY_NAMES
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    logger.info("Phase 4 feature ranking and tiering complete")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase4/features.yaml", help="Path to the Phase 4 features config YAML"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
