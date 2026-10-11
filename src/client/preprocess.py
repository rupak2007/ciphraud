"""Client-side plaintext preprocessing (`docs/architecture.md` Sec.6: preprocessing stays client-side and plaintext).

Pure numpy, importable everywhere. `preprocessing.json` (written next to `client.zip` by `src/fhe/deploy.py`) says what to
do for one deployed model: LR and the MLP standardize with the scaler fitted on the TRAIN split only (the same
`standardize_features` Phase 8/9 used, at the same dtype, so the quantized integers are bit-identical to the recorded
ones); XGBoost consumes the raw features unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.fhe.handoff import standardize_features


def load_preprocessing(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_preprocessing(spec: dict[str, Any], X_raw: np.ndarray) -> np.ndarray:
    """Raw feature rows -> the array `model.quantize_input` expects."""
    X_raw = np.asarray(X_raw)
    if X_raw.ndim != 2:
        raise ValueError(f"expected a 2-D array of raw feature rows, got shape {X_raw.shape}")
    if spec.get("n_features") is not None and X_raw.shape[1] != int(spec["n_features"]):
        raise ValueError(f"expected {spec['n_features']} features, got {X_raw.shape[1]}")
    kind = spec["kind"]
    if kind == "none":
        return X_raw
    if kind == "standardize":
        standardized = standardize_features({"scaler_mean": spec["scaler_mean"], "scaler_scale": spec["scaler_scale"]}, X_raw)
        return standardized.astype(spec["dtype"]) if spec.get("dtype") else standardized
    raise ValueError(f"unknown preprocessing kind {kind!r}")
