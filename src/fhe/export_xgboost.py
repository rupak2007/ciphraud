"""Phase 6 FHE XGBoost -- Windows-side export (`docs/plan.md` Phase 6).

Run as:
    python -m src.fhe.export_xgboost --config configs/phase6/xgb_poc.yaml

Runs in the Windows `.venv` and never invokes Concrete-ML. For each configured
tier it produces the ONLY thing the WSL side depends on: a verified
inference-time XGBoost booster (native JSON) plus the same pickle-free
`.npz` + manifest handoff Phase 5 uses (`src/fhe/handoff.py`, reused
unchanged).

Why the handoff is simpler than Phase 5's LR one: XGBoost's native JSON
model format is portable across XGBoost versions, so no numpy
reimplementation of `predict_proba` is needed. T0
(`src/fhe/validate/correctness.py`) still measures cross-version prediction
agreement empirically rather than assuming it.

Why the exported booster is SLICED: each committed Phase 4 booster stores
`best_iteration + 1 + early_stopping_rounds` trees, but the plaintext model
only uses the first `best_iteration + 1` at inference (XGBoost's
`predict_proba` applies that range automatically). Concrete-ML converts every
stored tree (docs/fhe_xgboost.md), so exporting the full booster would compile
a DIFFERENT model than the reference. Slicing is not a reduction: this module
asserts the exported booster reproduces the committed reference bit-for-bit,
and that its tree count equals Phase 4's committed `selected_n_estimators`.

The test partition (`X_test`/`y_test`) is loaded but never referenced except
`len()`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.features.tiers import load_tiers
from src.fhe.handoff import save_handoff, sha256_file
from src.logging_setup import get_logger
from src.train.data import load_phase2_features

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "PyYAML"]
EXPECTED_OBJECTIVE = "binary:logistic"


class Phase4XGBIntegrityError(Exception):
    """Raised when a committed Phase 4 tier XGBoost model doesn't match its own committed metrics."""


def _verify_phase4_provenance(phase4_config: dict[str, Any]) -> dict[str, Any]:
    """Phase 4's config must be unchanged since its committed run; returns the committed metrics."""
    output_dir = PROJECT_ROOT / phase4_config["output"]["dir"]
    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "provenance.json"
    missing = [p for p in (metrics_path, provenance_path) if not p.exists()]
    if missing:
        raise Phase4XGBIntegrityError(f"Phase 4 artifacts missing: {[str(p) for p in missing]}")

    recorded_hash = json.loads(provenance_path.read_text())["config_hash"]
    current_hash = data_provenance.compute_config_hash(phase4_config)
    if recorded_hash != current_hash:
        raise Phase4XGBIntegrityError(
            f"Phase 4 config_hash mismatch (committed={recorded_hash}, current={current_hash}) -- "
            f"configs/phase4/features.yaml changed since the committed run"
        )
    return json.loads(metrics_path.read_text())


def booster_objective(booster: xgb.Booster) -> str:
    return json.loads(booster.save_config())["learner"]["objective"]["name"]


def inference_booster(model: xgb.XGBClassifier) -> tuple[xgb.Booster, int, int]:
    """Slice to exactly the trees `predict_proba` uses. Returns (sliced, n_stored, best_iteration)."""
    booster = model.get_booster()
    best_iteration_attr = booster.attr("best_iteration")
    if best_iteration_attr is None:
        raise Phase4XGBIntegrityError("Booster has no best_iteration attribute; expected an early-stopped model")
    best_iteration = int(best_iteration_attr)
    n_stored = booster.num_boosted_rounds()
    return booster[0 : best_iteration + 1], n_stored, best_iteration


def export_tier(
    tier: str,
    tier_columns: list[str],
    features: Any,
    committed_metrics: dict[str, Any],
    phase4_models_dir: Path,
    handoff_dir: Path,
    tier_output_dir: Path,
    membership_hash: str,
    integrity_tolerance: float,
) -> dict[str, Any]:
    if tier not in committed_metrics["tiers"]:
        raise Phase4XGBIntegrityError(f"Tier {tier!r} not present in Phase 4's committed metrics.json")
    committed_xgb = committed_metrics["tiers"][tier]["xgboost"]

    model_path = phase4_models_dir / tier / "xgboost.json"
    if not model_path.exists():
        raise Phase4XGBIntegrityError(f"Phase 4 tier model missing: {model_path} -- run Phase 4 first")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    objective = booster_objective(model.get_booster())
    if objective != EXPECTED_OBJECTIVE:
        raise Phase4XGBIntegrityError(f"{tier}: objective {objective!r}, expected {EXPECTED_OBJECTIVE!r}")

    X_val_f32 = features.X_val[tier_columns].to_numpy()
    X_val = X_val_f32.astype(np.float64)
    X_train = features.X_train[tier_columns].to_numpy().astype(np.float64)
    y_val = features.y_val.to_numpy()

    reference_val_prob = model.predict_proba(X_val)[:, 1]
    if not np.array_equal(reference_val_prob, model.predict_proba(X_val_f32)[:, 1]):
        raise Phase4XGBIntegrityError(f"{tier}: predictions differ between float32 and float64-upcast input")

    reloaded_pr_auc = float(average_precision_score(y_val, reference_val_prob))
    committed_pr_auc = committed_xgb["val_evaluation"]["pr_auc"]
    if not np.isclose(reloaded_pr_auc, committed_pr_auc, atol=integrity_tolerance, rtol=0):
        raise Phase4XGBIntegrityError(
            f"{tier}: reloaded XGBoost val PR-AUC {reloaded_pr_auc} != committed {committed_pr_auc} "
            f"(tolerance {integrity_tolerance})"
        )

    sliced, n_stored, best_iteration = inference_booster(model)
    n_inference = best_iteration + 1
    if n_inference != committed_xgb["selected_n_estimators"]:
        raise Phase4XGBIntegrityError(
            f"{tier}: best_iteration+1={n_inference} != committed selected_n_estimators "
            f"{committed_xgb['selected_n_estimators']}"
        )

    handoff_dir.mkdir(parents=True, exist_ok=True)
    booster_path = handoff_dir / f"xgb_{tier}.json"
    sliced.save_model(booster_path)

    exported = xgb.Booster()
    exported.load_model(booster_path)
    if exported.num_boosted_rounds() != n_inference:
        raise Phase4XGBIntegrityError(f"{tier}: exported booster has {exported.num_boosted_rounds()} trees")
    exported_prob = exported.inplace_predict(X_val)
    if not np.array_equal(exported_prob, reference_val_prob):
        diff = float(np.abs(exported_prob - reference_val_prob).max())
        raise Phase4XGBIntegrityError(f"{tier}: exported booster does not reproduce reference (max|diff|={diff})")

    threshold = committed_xgb["val_evaluation"]["f1_selected"]["selected_threshold"]
    manifest_extra = {
        "model_type": "xgboost",
        "tier": tier,
        "threshold": threshold,
        "objective": objective,
        "phase4_tiers_membership_hash": membership_hash,
        "phase4_model_sha256": sha256_file(model_path),
        "phase4_val_pr_auc": committed_pr_auc,
        "booster_path": booster_path.relative_to(PROJECT_ROOT).as_posix(),  # read back under WSL
        "booster_sha256": sha256_file(booster_path),
        "n_trees_stored_in_phase4_booster": n_stored,
        "n_trees_inference": n_inference,
        "best_iteration": best_iteration,
        "selected_max_depth": committed_xgb["selected_max_depth"],
        "selected_learning_rate": committed_xgb["selected_learning_rate"],
        "windows_git_commit": data_provenance.get_git_commit(),
        "windows_library_versions": data_provenance.get_library_versions(LIBRARY_NAMES),
    }
    save_handoff(
        handoff_dir / f"xgb_{tier}.npz",
        tier_output_dir / "handoff_manifest.json",
        X_train=X_train, X_val=X_val, y_val=y_val,
        val_transaction_ids=features.X_val.index.to_numpy(),
        reference_val_prob=reference_val_prob,
        columns=tier_columns, manifest_extra=manifest_extra,
    )

    summary = {
        "tier": tier,
        "n_features": len(tier_columns),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_trees_stored_in_phase4_booster": n_stored,
        "n_trees_inference": n_inference,
        "reloaded_val_pr_auc": reloaded_pr_auc,
        "threshold": threshold,
    }
    logger.info("Phase 6 tier exported", extra={"extra_fields": summary})
    return summary


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)

    logger.info("Loading Phase 2 features (sole source of model-ready data)")
    features = load_phase2_features(config["phase2_config"])

    phase4_config = load_config(config["phase4_config"])
    committed_metrics = _verify_phase4_provenance(phase4_config)
    tier_payload = load_tiers(PROJECT_ROOT / phase4_config["output"]["dir"] / "tiers.json")

    output_dir = PROJECT_ROOT / config["output"]["dir"]
    tier_summaries = []
    for tier in config["tiers"]:
        tier_summaries.append(
            export_tier(
                tier=tier,
                tier_columns=tier_payload["tiers"][tier],
                features=features,
                committed_metrics=committed_metrics,
                phase4_models_dir=PROJECT_ROOT / phase4_config["output"]["models_dir"],
                handoff_dir=PROJECT_ROOT / config["output"]["handoff_dir"],
                tier_output_dir=output_dir / tier,
                membership_hash=tier_payload["membership_hash"],
                integrity_tolerance=config["phase4_integrity_tolerance"],
            )
        )

    summary = {"tiers": tier_summaries, "n_test": len(features.X_test), "test_partition_touched": False}
    logger.info("Phase 6 export complete", extra={"extra_fields": {"n_tiers": len(tier_summaries)}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6/xgb_poc.yaml", help="Path to the Phase 6 config YAML")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
