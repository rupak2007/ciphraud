"""Concrete-ML compilation for the Phase 5 Logistic Regression PoC.

WSL2/Linux only -- Concrete-ML has no Windows wheels (`docs/environment.md`).
Every function here takes plain numpy arrays and a `concrete.ml.sklearn`
model; nothing here reads a config or a file path, so it stays unit-testable
against a tiny synthetic model without any real data.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


def build_concrete_lr(
    params: dict[str, Any], X_calibration: np.ndarray, n_bits: int
) -> ConcreteLogisticRegression:
    """Reconstruct a plain sklearn LogisticRegression from the exported
    handoff parameters (`src/fhe/handoff.py::extract_lr_pipeline_params`),
    scaled coefficients already baked in -- so Concrete-ML quantizes/
    compiles a single linear model, no separate scaler step in the circuit
    (the scaler ran client-side, plaintext, in `src/fhe/export.py`, per
    `docs/architecture.md` Sec.6).

    `from_sklearn_model` needs a real, FIT sklearn estimator (it reads
    `coef_`/`intercept_`/`classes_`/`n_features_in_`), so this builds one
    by directly assigning the exported parameters rather than re-fitting --
    the whole point of the parameter handoff is to avoid depending on
    scikit-learn's pickle format across the two environments' different
    scikit-learn versions (`src/fhe/handoff.py` module docstring).

    `X_calibration` (the SCALED train partition, per-feature quantization
    range derived from it) must be the real calibration set, never the test
    partition or a placeholder -- `from_sklearn_model`'s `n_bits`-width
    quantization grid is calibrated directly from this array's per-feature
    min/max (`docs/architecture.md` Sec.5).
    """
    coef = np.asarray(params["lr_coef"], dtype=np.float64)
    intercept = np.asarray(params["lr_intercept"], dtype=np.float64)
    classes = np.asarray(params["lr_classes"])

    sk_model = SklearnLogisticRegression(C=params["lr_C"])
    sk_model.coef_ = coef
    sk_model.intercept_ = intercept
    sk_model.classes_ = classes
    sk_model.n_features_in_ = coef.shape[1]

    return ConcreteLogisticRegression.from_sklearn_model(sk_model, X_calibration, n_bits=n_bits)


def compile_model(cml_model: ConcreteLogisticRegression, X_calibration: np.ndarray) -> tuple[Any, float]:
    """Compile against a calibration inputset (`docs/architecture.md` Sec.5:
    "calibration dataset ... disjoint from the final test set"). Returns
    the circuit and the wall-clock compile time in seconds -- an
    observational measurement only (Phase 7 owns real benchmarking with
    repeated trials)."""
    t0 = time.perf_counter()
    circuit = cml_model.compile(X_calibration)
    compile_seconds = time.perf_counter() - t0
    return circuit, compile_seconds


def circuit_stats(circuit: Any) -> dict[str, Any]:
    """Every circuit-level statistic `docs/plan.md`/`CLAUDE.md` ask to be
    logged: PBS count (the dominant FHE cost driver for nonlinear ops --
    expected to be 0 here, since the sigmoid/threshold run client-side
    post-decryption, not inside the circuit), key/ciphertext sizes, and
    `p_error`. A plain dict of already-computed circuit properties -- no
    extra compilation or execution triggered by calling this."""
    return {
        "programmable_bootstrap_count": int(circuit.programmable_bootstrap_count),
        "p_error": float(circuit.p_error),
        "global_p_error": float(circuit.global_p_error),
        "complexity": float(circuit.complexity),
        "size_of_secret_keys_bytes": int(circuit.size_of_secret_keys),
        "size_of_bootstrap_keys_bytes": int(circuit.size_of_bootstrap_keys),
        "size_of_keyswitch_keys_bytes": int(circuit.size_of_keyswitch_keys),
        "size_of_inputs_bytes": int(circuit.size_of_inputs),
        "size_of_outputs_bytes": int(circuit.size_of_outputs),
    }
