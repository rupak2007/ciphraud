"""Phase 5 FHE PoC -- Windows-side export (`docs/plan.md` Phase 5).

Run as:
    python -m src.fhe.export --config configs/phase5/lr_poc.yaml

Concrete-ML only runs under WSL2/Linux (`docs/environment.md`); this script
runs in the Windows `.venv` and produces the ONLY thing the WSL side is
allowed to depend on for the plaintext model and data: the parameter-level
handoff in `src/fhe/handoff.py`. It never invokes Concrete-ML itself.

Orchestrates, in order:
  1. Load Phase 2 features (`src/train/data.py::load_phase2_features`) --
     the sole source of model-ready data, exactly as Phase 3/4.
  2. Load the committed Phase 4 tiers (`src/features/tiers.py::load_tiers`,
     hash-verified) and select the configured tier's column list, IN THE
     ORDER `tiers.json` RECORDS -- column order is part of the model
     definition (docs/features.md Sec.6.1's finding), not cosmetic.
  3. Verify the committed Phase 4 top_20 LR model reproduces its own
     committed val PR-AUC after a plain reload (no re-fit) -- the same
     "reload the saved artifact" integrity pattern
     `src/features/pipeline.py::_verify_phase3_integrity` uses for Phase 3.
  4. Extract that model's exact scaler+LR parameters
     (`src/fhe/handoff.py::extract_lr_pipeline_params`) and write the
     calibration (train) / correctness-check (val) matrices, in the SAME
     tier column order, to a pickle-free `.npz` + JSON manifest.

**The test partition (`X_test`/`y_test`) is loaded but never referenced in
this file except `len()`** -- the same deliberate property Phase 3/4 have,
verified the same way (`tests/test_fhe_export.py::test_test_partition_never_touched`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.handoff import extract_lr_pipeline_params, save_handoff, sha256_file
from src.features.tiers import load_tiers
from src.logging_setup import get_logger
from src.train.data import load_phase2_features

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "PyYAML"]


class Phase4IntegrityError(Exception):
    """Raised when the committed Phase 4 tier model doesn't match its own committed metrics."""


def _verify_phase4_tier_integrity(
    phase4_config: dict[str, Any], tier: str, tier_columns: list[str], features, tolerance: float
) -> dict[str, Any]:
    """Reload the COMMITTED Phase 4 tier LR model (no re-fit) and confirm
    (a) Phase 4's config hasn't changed since its committed run, and (b) it
    still reproduces Phase 4's own committed `metrics.json` val PR-AUC --
    ties config <-> model file <-> committed metrics <-> current Phase 2
    feature matrix together before anything is exported, exactly the two
    checks `src/features/pipeline.py::_verify_phase3_integrity` performs
    for Phase 3 relative to Phase 4."""
    output_dir = PROJECT_ROOT / phase4_config["output"]["dir"]
    models_dir = PROJECT_ROOT / phase4_config["output"]["models_dir"]
    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    lr_path = models_dir / tier / "logistic_regression.joblib"

    missing = [p for p in (metrics_path, provenance_path, lr_path) if not p.exists()]
    if missing:
        raise Phase4IntegrityError(
            f"Phase 4 tier artifacts missing: {[str(p) for p in missing]} -- run Phase 4 first"
        )

    recorded_hash = json.loads(provenance_path.read_text())["config_hash"]
    current_hash = data_provenance.compute_config_hash(phase4_config)
    if recorded_hash != current_hash:
        raise Phase4IntegrityError(
            f"Phase 4 config_hash mismatch (committed={recorded_hash}, current={current_hash}) -- "
            f"configs/phase4/features.yaml changed since the committed run; re-run Phase 4 first"
        )

    committed_metrics = json.loads(metrics_path.read_text())
    if tier not in committed_metrics["tiers"]:
        raise Phase4IntegrityError(f"Tier {tier!r} not present in Phase 4's committed metrics.json")

    model = joblib.load(lr_path)
    X_val = features.X_val[tier_columns].to_numpy()
    y_val = features.y_val.to_numpy()
    reloaded_pr_auc = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
    committed_pr_auc = committed_metrics["tiers"][tier]["logistic_regression"]["val_evaluation"]["pr_auc"]

    if not np.isclose(reloaded_pr_auc, committed_pr_auc, atol=tolerance, rtol=0):
        raise Phase4IntegrityError(
            f"Reloaded Phase 4 {tier} LR model's val PR-AUC ({reloaded_pr_auc}) does not match "
            f"committed metrics.json ({committed_pr_auc}) within tolerance {tolerance}"
        )

    logger.info(
        "Phase 4 tier integrity verified",
        extra={"extra_fields": {"tier": tier, "reloaded_pr_auc": reloaded_pr_auc}},
    )
    return {"model": model, "committed_metrics": committed_metrics}


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    tier = config["tier"]

    logger.info("Loading Phase 2 features (sole source of model-ready data)")
    features = load_phase2_features(config["phase2_config"])
    # TEST PARTITION: features.X_test/features.y_test exist but are never
    # referenced below except len() in the returned summary -- the test
    # partition stays untouched, per this project's standing rule.

    logger.info("Loading committed Phase 4 tiers (hash-verified)")
    tiers_path = PROJECT_ROOT / load_config(config["phase4_config"])["output"]["dir"] / "tiers.json"
    tier_payload = load_tiers(tiers_path)
    tier_columns = tier_payload["tiers"][tier]
    logger.info("Selected tier", extra={"extra_fields": {"tier": tier, "n_features": len(tier_columns)}})

    phase4_config = load_config(config["phase4_config"])
    integrity = _verify_phase4_tier_integrity(
        phase4_config, tier, tier_columns, features, config["phase4_integrity_tolerance"]
    )
    model = integrity["model"]
    threshold = integrity["committed_metrics"]["tiers"][tier]["logistic_regression"]["val_evaluation"][
        "f1_selected"
    ]["selected_threshold"]

    logger.info("Extracting exact scaler+LR parameters")
    params = extract_lr_pipeline_params(model)

    # Upcast to float64 HERE, before computing reference_val_prob -- a real,
    # measured finding (docs/fhe_poc.md Sec.6.1), not a style choice:
    # sklearn's StandardScaler.transform preserves its INPUT's dtype
    # (confirmed empirically: transform(float32_X).dtype == float32), so
    # calling predict_proba on the raw float32 Phase 2 matrix evaluates the
    # scaler step at float32 precision even though mean_/scale_ are stored
    # as float64. That is a ~1e-7 (probability-level) / ~1e-6 (logit-level)
    # discrepancy against handoff.rebuild_pipeline_predict_proba's pure
    # float64 arithmetic -- not a transfer bug, but it would blow through a
    # strict T0 tolerance for a reason that has nothing to do with the
    # cross-environment handoff T0 exists to check. Upcasting here makes
    # "the reference" unambiguously float64-precise BEFORE it's computed, so
    # it matches the WSL-side rebuild by construction. This makes
    # reference_val_prob differ from Phase 4's own committed (float32-
    # precision) val probabilities by ~1e-7 -- immaterial to any reported
    # PR-AUC digit, and recorded as a deviation, not silently absorbed.
    X_train_tier = features.X_train[tier_columns].to_numpy().astype(np.float64)
    X_val_tier = features.X_val[tier_columns].to_numpy().astype(np.float64)
    y_val = features.y_val.to_numpy()
    val_transaction_ids = features.X_val.index.to_numpy()
    reference_val_prob = model.predict_proba(X_val_tier)[:, 1]

    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = PROJECT_ROOT / config["output"]["params_path"]
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")

    npz_path = PROJECT_ROOT / config["output"]["handoff_npz"]
    manifest_path = PROJECT_ROOT / config["output"]["handoff_manifest"]
    manifest_extra = {
        "tier": tier,
        "threshold": threshold,
        "phase4_tiers_membership_hash": tier_payload["membership_hash"],
        "phase4_model_sha256": sha256_file(
            PROJECT_ROOT / phase4_config["output"]["models_dir"] / tier / "logistic_regression.joblib"
        ),
        "params_path": str(config["output"]["params_path"]),
        "params_sha256": sha256_file(params_path),
        "windows_git_commit": data_provenance.get_git_commit(),
        "windows_library_versions": data_provenance.get_library_versions(LIBRARY_NAMES),
    }
    save_handoff(
        npz_path, manifest_path,
        X_train=X_train_tier, X_val=X_val_tier, y_val=y_val,
        val_transaction_ids=val_transaction_ids, reference_val_prob=reference_val_prob,
        columns=tier_columns, manifest_extra=manifest_extra,
    )

    summary = {
        "tier": tier,
        "n_features": len(tier_columns),
        "n_train": int(X_train_tier.shape[0]),
        "n_val": int(X_val_tier.shape[0]),
        "n_test": len(features.X_test),
        "threshold": threshold,
        "test_partition_touched": False,
    }
    logger.info("Phase 5 export complete", extra={"extra_fields": summary})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase5/lr_poc.yaml", help="Path to the Phase 5 PoC config YAML")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
