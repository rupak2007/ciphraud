"""Layered plaintext-vs-FHE correctness checks for the Phase 5 PoC.

Each check isolates one specific source of error, so a failure has exactly
one attributable cause (`docs/fhe_poc.md` Sec.5):

  T0 (transfer)      WSL-reconstructed float model  vs. Windows reference
  T3 (quantization)  Concrete-ML clear-quantized     vs. float model
  T2 (simulation)    Concrete-ML fhe="simulate"      vs. fhe="disable"
  T1 (encryption)    real encrypt->run->decrypt      vs. fhe="simulate"

T1/T2 require EXACT agreement (`docs/plan.md` Phase 5's "validate against
plaintext output", made concrete by this project's own approved tolerances)
-- there is no probabilistic bootstrapping in a circuit with
`programmable_bootstrap_count == 0` (a linear model's affine arithmetic, per
`src/fhe/compile/linear.py`), so "simulate" and a real encrypt/run/decrypt
round trip are expected to be bit-for-bit identical, not merely close.

Every function here is a pure function of arrays it's given -- no I/O, no
Concrete-ML import required in this module itself, so it's testable with
plain numpy on either environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.train.metrics import metrics_at_threshold, threshold_independent_metrics


def t0_transfer_check(rebuilt_prob: np.ndarray, reference_prob: np.ndarray, max_abs_diff: float) -> dict[str, Any]:
    """Rebuilt-from-params float probabilities vs. the Windows-computed
    reference (`src/fhe/handoff.py`). Isolates cross-environment transfer
    error from everything else -- if this fails, the handoff itself is
    broken, before Concrete-ML is even involved."""
    diff = np.abs(np.asarray(rebuilt_prob) - np.asarray(reference_prob))
    result = {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "tolerance": max_abs_diff,
        "passed": bool(diff.max() <= max_abs_diff),
    }
    return result


def t3_quantization_check(
    quantized_prob: np.ndarray,
    float_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    min_decision_agreement: float,
    max_pr_auc_drop: float,
) -> dict[str, Any]:
    """Concrete-ML's clear-quantized model (`fhe="disable"`) vs. the float
    plaintext model, on the FULL val partition. Isolates quantization
    error (n_bits=8) from FHE execution entirely -- no encryption is
    involved in either side of this comparison.

    Gates (both approved, both required to pass):
      - decision agreement at the tier's val-selected threshold >= min_decision_agreement
      - PR-AUC drop (float PR-AUC - quantized PR-AUC) <= max_pr_auc_drop
    """
    quantized_prob = np.asarray(quantized_prob)
    float_prob = np.asarray(float_prob)
    y_true = np.asarray(y_true)

    quantized_decision = (quantized_prob >= threshold).astype(int)
    float_decision = (float_prob >= threshold).astype(int)
    decision_agreement = float(np.mean(quantized_decision == float_decision))

    quantized_metrics = threshold_independent_metrics(y_true, quantized_prob)
    float_metrics = threshold_independent_metrics(y_true, float_prob)
    pr_auc_drop = float(float_metrics["pr_auc"] - quantized_metrics["pr_auc"])

    abs_diff = np.abs(quantized_prob - float_prob)
    return {
        "decision_agreement": decision_agreement,
        "min_decision_agreement": min_decision_agreement,
        "pr_auc_drop": pr_auc_drop,
        "max_pr_auc_drop": max_pr_auc_drop,
        "quantized_pr_auc": quantized_metrics["pr_auc"],
        "float_pr_auc": float_metrics["pr_auc"],
        "quantized_roc_auc": quantized_metrics["roc_auc"],
        "float_roc_auc": float_metrics["roc_auc"],
        "max_abs_prob_diff": float(abs_diff.max()),
        "mean_abs_prob_diff": float(abs_diff.mean()),
        "quantized_confusion_matrix": metrics_at_threshold(y_true, quantized_prob, threshold)["confusion_matrix"],
        "float_confusion_matrix": metrics_at_threshold(y_true, float_prob, threshold)["confusion_matrix"],
        "passed": bool(decision_agreement >= min_decision_agreement and pr_auc_drop <= max_pr_auc_drop),
    }


def t2_simulation_check(simulate_prob: np.ndarray, disable_prob: np.ndarray) -> dict[str, Any]:
    """`fhe="simulate"` vs. `fhe="disable"`, on the FULL val partition.
    Isolates any effect of the compiled circuit itself (accumulator
    overflow, calibration-range clipping, `p_error`) from quantization,
    which both sides already share. Required to be EXACT."""
    simulate_prob = np.asarray(simulate_prob)
    disable_prob = np.asarray(disable_prob)
    exact_match = bool(np.array_equal(simulate_prob, disable_prob))
    diff = np.abs(simulate_prob - disable_prob)
    return {
        "exact_match": exact_match,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "n_mismatched": int(np.sum(~np.isclose(simulate_prob, disable_prob, atol=0, rtol=0))),
        "n_total": int(simulate_prob.shape[0]),
        "passed": exact_match,
    }


def t1_execution_check(decrypted_prob: np.ndarray, simulate_prob: np.ndarray) -> dict[str, Any]:
    """Real encrypt->run->decrypt round-trip output vs. `fhe="simulate"`,
    on the seeded execute sample. Isolates actual FHE execution error
    (as opposed to the simulator's noiseless model of it). Required to be
    EXACT -- this is `docs/plan.md` Phase 5's literal exit criterion:
    "at least one full encrypt->infer->decrypt round trip completes
    correctly", made falsifiable by this exact comparison."""
    decrypted_prob = np.asarray(decrypted_prob)
    simulate_prob = np.asarray(simulate_prob)
    exact_match = bool(np.array_equal(decrypted_prob, simulate_prob))
    diff = np.abs(decrypted_prob - simulate_prob)
    return {
        "exact_match": exact_match,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "n_rows_executed": int(decrypted_prob.shape[0]),
        "passed": exact_match,
    }
