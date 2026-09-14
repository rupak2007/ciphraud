"""Concrete-ML compilation for Phase 6 XGBoost (`docs/plan.md` Phase 6).

WSL2/Linux only -- Concrete-ML has no Windows wheels (`docs/environment.md`).
Compilation timing and circuit statistics are model-agnostic and come from
`src/fhe/compile/linear.py` (`compile_model`, `circuit_stats`) unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier

EXPECTED_OBJECTIVE = "binary:logistic"


class TreeCompileError(Exception):
    """Raised when a booster can't be turned into the expected Concrete-ML model."""


def load_inference_classifier(booster_path: Path) -> xgb.XGBClassifier:
    """Load the exported inference-time booster (`src/fhe/export_xgboost.py`) as a
    plain `xgboost.XGBClassifier` that `from_sklearn_model` accepts."""
    clf = xgb.XGBClassifier()
    clf.load_model(booster_path)
    objective = json.loads(clf.get_booster().save_config())["learner"]["objective"]["name"]
    if objective != EXPECTED_OBJECTIVE:
        raise TreeCompileError(f"{booster_path}: objective {objective!r}, expected {EXPECTED_OBJECTIVE!r}")
    # xgboost 1.6.2's load_model doesn't restore n_classes_ (only fit() sets it), and
    # Hummingbird's converter requires it. binary:logistic is two classes, checked above.
    clf.n_classes_ = 2
    return clf


def build_concrete_xgb(clf: xgb.XGBClassifier, X_calibration: np.ndarray, n_bits: int) -> ConcreteXGBClassifier:
    """`n_bits` sets input-feature precision; Concrete-ML derives the intermediate
    (comparison/leaf) bit-widths from `X_calibration` (`docs/architecture.md` Sec.5).
    `X_calibration` must come from the train partition only.

    Concrete-ML converts every tree stored in the booster, ignoring
    `best_iteration` -- which is why the exported booster is already sliced
    to the inference-time trees."""
    return ConcreteXGBClassifier.from_sklearn_model(clf, X_calibration, n_bits=n_bits)


def tree_stats(clf: xgb.XGBClassifier, cml_model: ConcreteXGBClassifier, circuit: Any) -> dict[str, Any]:
    """Tree-specific values `docs/architecture.md` Sec.5 asks to log per configuration."""
    booster = clf.get_booster()
    return {
        "n_trees": int(booster.num_boosted_rounds()),
        "n_features": int(clf.n_features_in_),
        "n_bits_inputs": sorted({int(q.n_bits) for q in cml_model.input_quantizers}),
        "n_bits_output": int(cml_model.output_quantizers[0].n_bits),
        "max_integer_bit_width": int(circuit.graph.maximum_integer_bit_width()),
    }
