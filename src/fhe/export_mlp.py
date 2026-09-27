"""Phase 9 (quantized MLP) -- Windows-side export (`docs/plan.md` Phase 9).

Run as:
    python -m src.fhe.export_mlp --config configs/phase9/mlp_export.yaml

Runs in the Windows `.venv` and never invokes Concrete-ML. For each configured tier it writes the
parameter-level handoff (`src/fhe/handoff.py`) the WSL side trains and compiles the MLP from:

  * `X_train`, `y_train`, `X_val`, `y_val`, `val_transaction_ids` (pickle-free `.npz`, hash-verified on load);
  * a train-only `StandardScaler`'s mean/scale as plain JSON (the MLP, like the LR, is trained and compiled on
    STANDARDIZED inputs; the scaler stays client-side and plaintext, `docs/architecture.md` Sec.6). The scaler
    is fit on `X_train` only -- no validation or test statistics.

Tier membership and column ORDER come from the committed, hash-verified Phase 4 `tiers.json`, exactly as in
Phases 5-8, so the matrices equal the Phase 8 LR handoff of the same tier. Unlike Phases 5-8 no Windows-side
model exists to verify (the MLP is trained in WSL), so there is no reference probability and no
Phase-4-model integrity check here; the integrity of the *data* is what the tests check.

**The test partition (`X_test`/`y_test`) is loaded by `load_phase2_features` but never referenced in this file
except `len()`** (tests/test_fhe_export_mlp.py::test_test_partition_never_touched).
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.features.tiers import load_tiers
from src.fhe.handoff import save_handoff, sha256_file
from src.logging_setup import get_logger
from src.train.data import load_phase2_features

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "PyYAML"]


def fit_train_only_scaler(X_train: np.ndarray) -> dict[str, list[float]]:
    """Mean and scale of a `StandardScaler` fit on the TRAIN matrix only (population std, ddof=0, and a zero-variance
    column gets scale 1.0 -- the same conventions as scikit-learn's `StandardScaler`, so the arithmetic matches the
    LR pipeline's scaler step)."""
    X = np.asarray(X_train, dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return {"scaler_mean": mean.tolist(), "scaler_scale": scale.tolist()}


def export_tier(config: dict[str, Any], features: Any, tier_payload: dict[str, Any], tier: str) -> dict[str, Any]:
    tier_columns = tier_payload["tiers"][tier]
    X_train = features.X_train[tier_columns].to_numpy().astype(np.float64)
    X_val = features.X_val[tier_columns].to_numpy().astype(np.float64)
    y_train = features.y_train.to_numpy()
    y_val = features.y_val.to_numpy()
    val_transaction_ids = features.X_val.index.to_numpy()

    if np.isnan(X_train).any() or np.isnan(X_val).any():
        raise ValueError(f"tier {tier}: NaN in the model-ready matrices (Phase 2 imputes; none are expected)")
    if set(np.unique(y_train)) - {0, 1} or set(np.unique(y_val)) - {0, 1}:
        raise ValueError(f"tier {tier}: labels are not binary")

    out_dir = PROJECT_ROOT / config["output"]["dir"] / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = out_dir / f"mlp_{tier}_scaler.json"
    scaler_path.write_text(json.dumps(fit_train_only_scaler(X_train), indent=2, sort_keys=True), encoding="utf-8")

    npz_path = PROJECT_ROOT / config["output"]["handoff_dir"] / f"mlp_{tier}.npz"
    manifest_path = out_dir / "handoff_manifest.json"
    manifest_extra = {
        "tier": tier, "model": "mlp",
        "phase4_tiers_membership_hash": tier_payload["membership_hash"],
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scaler_sha256": sha256_file(scaler_path),
        "scaler_fit_on": "X_train only",
        "train_fraud_rate": float(y_train.mean()), "val_fraud_rate": float(y_val.mean()),
        "test_partition_touched": False,
        "windows_git_commit": data_provenance.get_git_commit(),
        "windows_library_versions": data_provenance.get_library_versions(LIBRARY_NAMES),
    }
    save_handoff(
        npz_path, manifest_path, X_train=X_train, X_val=X_val, y_val=y_val, val_transaction_ids=val_transaction_ids,
        reference_val_prob=None, columns=tier_columns, manifest_extra=manifest_extra, y_train=y_train,
    )
    summary = {
        "tier": tier, "n_features": len(tier_columns), "n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]),
        "train_fraud_rate": float(y_train.mean()), "val_fraud_rate": float(y_val.mean()), "test_partition_touched": False,
    }
    logger.info("Phase 9 MLP handoff exported", extra={"extra_fields": summary})
    return summary


def run(config_path: str) -> list[dict[str, Any]]:
    config = load_config(config_path)
    logger.info("Loading Phase 2 features (sole source of model-ready data)")
    features = load_phase2_features(config["phase2_config"])
    # TEST PARTITION: features.X_test / features.y_test are never referenced below (not even len()).
    tiers_path = PROJECT_ROOT / load_config(config["phase4_config"])["output"]["dir"] / "tiers.json"
    tier_payload = load_tiers(tiers_path)
    return [export_tier(config, features, tier_payload, tier) for tier in config["tiers"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase9/mlp_export.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
