"""Tests for src/fhe/validate/correctness.py -- pure numpy, no Concrete-ML
import, so these run in both the Windows and WSL environments."""

import numpy as np

from src.fhe.validate.correctness import (
    t0_transfer_check,
    t1_execution_check,
    t2_simulation_check,
    t3_quantization_check,
)


def test_t0_transfer_check_passes_within_tolerance():
    ref = np.array([0.1, 0.5, 0.9])
    rebuilt = ref + 1e-10
    result = t0_transfer_check(rebuilt, ref, max_abs_diff=1e-9)
    assert result["passed"] is True
    assert result["max_abs_diff"] < 1e-9


def test_t0_transfer_check_fails_beyond_tolerance():
    ref = np.array([0.1, 0.5, 0.9])
    rebuilt = ref + 1e-3
    result = t0_transfer_check(rebuilt, ref, max_abs_diff=1e-9)
    assert result["passed"] is False
    assert result["max_abs_diff"] > 1e-9


def test_t2_simulation_check_exact_match_passes():
    prob = np.array([0.01, 0.4, 0.99, 0.5])
    result = t2_simulation_check(prob.copy(), prob.copy())
    assert result["passed"] is True
    assert result["exact_match"] is True
    assert result["n_mismatched"] == 0


def test_t2_simulation_check_any_difference_fails():
    a = np.array([0.01, 0.4, 0.99, 0.5])
    b = a.copy()
    b[2] += 1e-12  # even a tiny difference must fail -- EXACT is required
    result = t2_simulation_check(a, b)
    assert result["passed"] is False
    assert result["exact_match"] is False
    assert result["n_mismatched"] == 1


def test_t1_execution_check_exact_match_passes():
    prob = np.array([0.2, 0.8, 0.05])
    result = t1_execution_check(prob.copy(), prob.copy())
    assert result["passed"] is True
    assert result["n_rows_executed"] == 3


def test_t1_execution_check_mismatch_fails():
    a = np.array([0.2, 0.8, 0.05])
    b = np.array([0.2, 0.8, 0.06])
    result = t1_execution_check(a, b)
    assert result["passed"] is False


def test_t3_quantization_check_identical_inputs_pass_perfectly():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=200)
    prob = rng.rand(200)
    result = t3_quantization_check(
        quantized_prob=prob, float_prob=prob, y_true=y_true, threshold=0.5,
        min_decision_agreement=0.99, max_pr_auc_drop=0.01,
    )
    assert result["passed"] is True
    assert result["decision_agreement"] == 1.0
    assert result["pr_auc_drop"] == 0.0


def test_t3_quantization_check_fails_on_low_agreement():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=200)
    float_prob = rng.rand(200)
    # Flip enough decisions to push agreement well below 99%.
    quantized_prob = 1.0 - float_prob
    result = t3_quantization_check(
        quantized_prob=quantized_prob, float_prob=float_prob, y_true=y_true, threshold=0.5,
        min_decision_agreement=0.99, max_pr_auc_drop=0.01,
    )
    assert result["passed"] is False
    assert result["decision_agreement"] < 0.99


def test_t3_quantization_check_fails_on_pr_auc_drop_even_with_high_agreement():
    """A model can agree with the float model on >=99% of THRESHOLD
    decisions while still having materially worse PR-AUC: PR-AUC depends
    on the FULL rank ordering of scores (like ROC-AUC), not just which
    side of one threshold each row falls on, so coarse-binning the
    probabilities (as a low-bit quantizer would) creates many ties that
    barely move threshold-decision agreement but measurably hurt ranking
    quality. Both gates are independently checked; neither alone is
    sufficient -- found by directly measuring a coarse-quantization
    construction until it exhibited exactly this combination (agreement
    >=0.99, drop >0.01), not assumed analytically."""
    rng = np.random.RandomState(0)
    n = 3000
    y_true = (rng.rand(n) < 0.1).astype(int)
    latent = rng.randn(n) + 1.3 * y_true
    float_prob = 1.0 / (1.0 + np.exp(-latent))
    bin_size = 0.03
    quantized_prob = np.clip(np.round(float_prob / bin_size) * bin_size, 0.001, 0.999)

    result = t3_quantization_check(
        quantized_prob=quantized_prob, float_prob=float_prob, y_true=y_true, threshold=0.5,
        min_decision_agreement=0.99, max_pr_auc_drop=0.01,
    )
    assert result["decision_agreement"] >= 0.99
    assert result["passed"] is False
    assert result["pr_auc_drop"] > 0.01
