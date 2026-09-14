"""Tests for src/fhe/validate/correctness.py -- pure numpy, no Concrete-ML
import, so these run in both the Windows and WSL environments."""

import numpy as np

import pytest

from src.fhe.validate.correctness import (
    integer_output_check,
    t0_transfer_check,
    t1_execution_check,
    t2_simulation_check,
    t3_quantization_check,
)


def _tree_outputs(n_rows=6, n_trees=5, seed=0):
    return np.random.RandomState(seed).randint(-20, 20, size=(n_rows, 1, n_trees)).astype(np.int64)


def test_integer_output_check_exact_match_passes():
    q = _tree_outputs()
    result = integer_output_check(q, q.copy())
    assert result["passed"] is True
    assert result["exact_integer_match"] is True
    assert result["n_rows"] == 6
    assert result["n_outputs_per_row"] == 5
    assert result["n_rows_with_integer_mismatch"] == 0


def test_integer_output_check_single_tree_off_by_one_fails():
    q = _tree_outputs()
    other = q.copy()
    other[3, 0, 2] += 1
    result = integer_output_check(other, q)
    assert result["passed"] is False
    assert result["n_rows_with_integer_mismatch"] == 1
    assert result["n_integer_outputs_mismatched"] == 1


def test_integer_output_check_ignores_memory_layout():
    """Same values, different strides (a non-contiguous view vs a contiguous copy)
    must compare equal -- the measured difference between Concrete-ML's batched
    `disable` path and its per-row `simulate` path."""
    non_contiguous = _tree_outputs(n_trees=14)[:, :, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    result = integer_output_check(np.ascontiguousarray(non_contiguous), non_contiguous)
    assert result["passed"] is True


def test_integer_output_check_reports_float_noise_without_failing():
    q = _tree_outputs()
    prob = np.array([0.1, 0.4, 0.5, 0.69, 0.71, 0.9])
    noisy = prob + np.array([0, 4e-16, 0, 0, 0, 0])
    result = integer_output_check(q, q.copy(), candidate_prob=noisy, reference_prob=prob, threshold=0.7)
    assert result["passed"] is True
    assert result["prob_n_not_bit_identical"] == 1
    assert result["prob_max_abs_diff"] < 1e-15
    assert result["n_decision_flips"] == 0


def test_integer_output_check_counts_decision_flips():
    q = _tree_outputs()
    prob = np.array([0.1, 0.4, 0.5, 0.69, 0.71, 0.9])
    moved = prob.copy()
    moved[3] = 0.72
    result = integer_output_check(q, q.copy(), candidate_prob=moved, reference_prob=prob, threshold=0.7)
    assert result["n_decision_flips"] == 1


def test_integer_output_check_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        integer_output_check(_tree_outputs(n_trees=5), _tree_outputs(n_trees=4))


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
