"""Cross-environment handoff for the Phase 5 FHE PoC (`docs/plan.md` Phase 5).

Concrete-ML only installs under WSL2/Linux (`docs/environment.md`), with its
own pinned scikit-learn (1.5.0) and numpy (1.26.4) -- materially older than
the Windows `.venv`'s (1.9.0 / 2.5.3), which is where the Phase 4 plaintext
LR model (`models/phase4_features/top_20/logistic_regression.joblib`) was
fit and pickled. `docs/environment.md` explicitly flagged that a model
pickled under one environment's scikit-learn is not guaranteed to
deserialize/behave identically under the other's, and deferred resolving
this to Phase 5.

Resolution: never unpickle across the boundary. `src/fhe/export.py` (Windows
side) extracts the fitted `StandardScaler` + `LogisticRegression`'s exact
numeric PARAMETERS (not the estimator objects) into plain JSON, and the
calibration/validation feature matrices into a pickle-free `.npz`
(`allow_pickle=False` on load, enforced below). This module -- imported by
BOTH `src/fhe/export.py` (Windows) and `src/fhe/poc.py` (WSL) -- is the only
place that reads/writes/verifies that handoff, and it depends on nothing but
numpy and the standard library, so it imports cleanly in both environments.

`rebuild_pipeline_predict_proba` reimplements the fitted pipeline's
`predict_proba` directly from those parameters using only `numpy`-level
arithmetic (no scikit-learn `Pipeline`/`StandardScaler`/`LogisticRegression`
object involved), so it is reproducible from the parameters alone
regardless of which scikit-learn version is installed -- this is what T0
(`src/fhe/validate/correctness.py`) verifies empirically, not assumed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class HandoffError(Exception):
    """Raised when the handoff manifest/npz fail integrity verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_lr_pipeline_params(pipeline: Any) -> dict[str, Any]:
    """Pull the exact numeric parameters out of a fitted
    `Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])`
    (the shape `src/train/logistic_regression.py` / Phase 4 produce).

    Every value is a plain Python float/int/list -- JSON round-trips them
    exactly (IEEE754 doubles have an exact decimal `repr` that `json`
    preserves), so no precision is lost crossing the environment boundary.
    """
    scaler = pipeline.named_steps["scaler"]
    lr = pipeline.named_steps["lr"]
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lr_coef": lr.coef_.tolist(),  # shape (1, n_features)
        "lr_intercept": lr.intercept_.tolist(),  # shape (1,)
        "lr_classes": lr.classes_.tolist(),
        "lr_C": float(lr.C),
        "n_features": int(scaler.mean_.shape[0]),
    }


def standardize_features(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """The pipeline's `StandardScaler` step, as a client-side plaintext
    transform: `(X - mean) / scale` in float64, from the exported parameters.

    The exported `lr_coef` are the coefficients of the model trained on
    STANDARDIZED features, so the compiled Concrete-ML LR must be built,
    calibrated, and evaluated on this function's output -- never on the raw
    handoff matrices (`docs/architecture.md` Sec.6: preprocessing stays
    client-side and plaintext). Feeding raw X to a model holding these
    coefficients silently evaluates sigmoid(w.x_raw + b) instead of
    sigmoid(w.(x-mean)/scale + b) (found in the Phase 8 audit; see
    docs/research.md, LR scaler erratum). Same arithmetic as
    `rebuild_pipeline_predict_proba`, so the two can never diverge."""
    mean = np.asarray(params["scaler_mean"], dtype=np.float64)
    scale = np.asarray(params["scaler_scale"], dtype=np.float64)
    return (np.asarray(X, dtype=np.float64) - mean) / scale


def rebuild_pipeline_predict_proba(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Pure-numpy reimplementation of `Pipeline.predict_proba(X)`, from
    parameters alone -- no scikit-learn estimator object involved, so this
    is exactly reproducible regardless of which scikit-learn version wrote
    the original model or is installed wherever this runs.

    Matches `sklearn.preprocessing.StandardScaler.transform` (subtract
    mean, divide by scale) followed by
    `sklearn.linear_model.LogisticRegression.predict_proba` for the binary
    case (sigmoid of the linear score, classes in `lr_classes` order).
    """
    mean = np.asarray(params["scaler_mean"], dtype=np.float64)
    scale = np.asarray(params["scaler_scale"], dtype=np.float64)
    coef = np.asarray(params["lr_coef"], dtype=np.float64).reshape(1, -1)
    intercept = np.asarray(params["lr_intercept"], dtype=np.float64).reshape(1)

    X_scaled = (np.asarray(X, dtype=np.float64) - mean) / scale
    logits = X_scaled @ coef.T + intercept  # shape (n_rows, 1)
    prob_positive = 1.0 / (1.0 + np.exp(-logits.ravel()))
    return np.column_stack([1.0 - prob_positive, prob_positive])


def save_handoff(
    npz_path: Path,
    manifest_path: Path,
    *,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_transaction_ids: np.ndarray,
    reference_val_prob: np.ndarray,
    columns: list[str],
    manifest_extra: dict[str, Any],
) -> None:
    """Write the `.npz` (numeric arrays only, no pickled objects) and the
    JSON manifest (hashes + shapes + provenance) that `load_handoff` (WSL
    side) verifies before touching anything else."""
    if npz_path.suffix != ".npz":
        raise HandoffError(f"npz_path must end in '.npz', got {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_path, "wb") as f:
        np.savez(
            f,
            X_train=np.asarray(X_train, dtype=np.float64),
            X_val=np.asarray(X_val, dtype=np.float64),
            y_val=np.asarray(y_val, dtype=np.int64),
            val_transaction_ids=np.asarray(val_transaction_ids, dtype=np.int64),
            reference_val_prob=np.asarray(reference_val_prob, dtype=np.float64),
        )
    manifest = {
        "npz_sha256": sha256_file(npz_path),
        "columns": list(columns),
        "n_train": int(np.asarray(X_train).shape[0]),
        "n_val": int(np.asarray(X_val).shape[0]),
        "n_features": len(columns),
        **manifest_extra,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_handoff(npz_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read + verify the manifest and npz. Raises `HandoffError` on any
    tamper/corruption/mismatch -- never proceeds on an unverified handoff.

    `allow_pickle=False` is load-bearing, not defensive boilerplate: it is
    what makes this npz safe to load in a different environment/library
    version than wrote it (no arbitrary object deserialization is possible).
    """
    if not npz_path.exists():
        raise HandoffError(f"Handoff npz not found: {npz_path}")
    if not manifest_path.exists():
        raise HandoffError(f"Handoff manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    actual_hash = sha256_file(npz_path)
    if actual_hash != manifest["npz_sha256"]:
        raise HandoffError(
            f"npz SHA-256 mismatch (manifest={manifest['npz_sha256']}, actual={actual_hash}) "
            f"-- the handoff file may have been modified after export"
        )

    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}

    if arrays["X_train"].shape[1] != manifest["n_features"]:
        raise HandoffError(
            f"X_train has {arrays['X_train'].shape[1]} columns, manifest declares "
            f"n_features={manifest['n_features']}"
        )
    if len(manifest["columns"]) != manifest["n_features"]:
        raise HandoffError("manifest 'columns' length does not match 'n_features'")

    return manifest, arrays
