import numpy as np
import pandas as pd
import pytest

from src.train.metrics import (
    error_analysis,
    full_evaluation,
    metrics_at_threshold,
    select_threshold,
    threshold_independent_metrics,
)

# A small, hand-checkable scenario: 10 rows, 3 positive, perfectly
# separable by a threshold of 0.5.
Y_TRUE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
Y_PROB_PERFECT = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0.2, 0.9, 0.8, 0.95])


def test_threshold_independent_metrics_perfect_separation_scores_high():
    result = threshold_independent_metrics(Y_TRUE, Y_PROB_PERFECT)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_threshold_independent_metrics_random_scores_near_baseline():
    rng = np.random.RandomState(0)
    y_true = (rng.rand(1000) < 0.1).astype(int)
    y_prob = rng.rand(1000)  # uninformative
    result = threshold_independent_metrics(y_true, y_prob)
    # PR-AUC baseline for random scores ~= positive rate; ROC-AUC ~= 0.5
    assert result["pr_auc"] == pytest.approx(0.1, abs=0.05)
    assert result["roc_auc"] == pytest.approx(0.5, abs=0.1)


def test_metrics_at_threshold_perfect_separation():
    result = metrics_at_threshold(Y_TRUE, Y_PROB_PERFECT, threshold=0.5)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["f2"] == pytest.approx(1.0)
    cm = result["confusion_matrix"]
    assert cm == {"tn": 7, "fp": 0, "fn": 0, "tp": 3}


def test_metrics_at_threshold_all_negative_predictions():
    y_prob = np.zeros_like(Y_PROB_PERFECT)
    result = metrics_at_threshold(Y_TRUE, y_prob, threshold=0.5)
    assert result["precision"] == 0.0  # zero_division=0, not an exception
    assert result["recall"] == 0.0
    cm = result["confusion_matrix"]
    assert cm["tp"] == 0 and cm["fn"] == 3


def test_metrics_at_threshold_confusion_matrix_sums_to_n():
    result = metrics_at_threshold(Y_TRUE, Y_PROB_PERFECT, threshold=0.5)
    cm = result["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(Y_TRUE)


def test_f2_weights_recall_more_than_f1():
    """Construct a case with higher recall than precision; F2 should exceed F1."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1])  # catches all positives, 2 false positives
    result = metrics_at_threshold(y_true, y_pred_prob, threshold=0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] < 1.0
    assert result["f2"] > result["f1"]


def test_select_threshold_f1_finds_reasonable_threshold():
    result = select_threshold(Y_TRUE, Y_PROB_PERFECT, metric="f1")
    assert result["metric_optimized"] == "f1"
    assert result["f1"] == pytest.approx(1.0)
    assert 0.3 < result["selected_threshold"] <= 0.9


def test_select_threshold_f2_can_differ_from_f1():
    """A case where a lower threshold (catching more positives at some
    precision cost) is F2-optimal but not F1-optimal."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.3, 0.9, 10)])
    f1_result = select_threshold(y_true, y_prob, metric="f1")
    f2_result = select_threshold(y_true, y_prob, metric="f2")
    # F2 favors recall -> should select a threshold <= the F1 threshold
    assert f2_result["selected_threshold"] <= f1_result["selected_threshold"] + 1e-9


def test_select_threshold_rejects_invalid_metric():
    with pytest.raises(ValueError, match="f1.*f2"):
        select_threshold(Y_TRUE, Y_PROB_PERFECT, metric="accuracy")


def test_full_evaluation_contains_all_required_fields():
    result = full_evaluation(Y_TRUE, Y_PROB_PERFECT)
    assert "pr_auc" in result
    assert "roc_auc" in result
    assert "f1_selected" in result
    assert "f2_selected" in result
    assert len(result["fixed_thresholds"]) == 3


def test_error_analysis_identifies_false_positives_and_negatives():
    index = pd.RangeIndex(10, name="TransactionID")
    y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    # row 0: false positive (true=0, prob=0.9 -> pred=1)
    # row 3: false negative (true=1, prob=0.1 -> pred=0)
    result = error_analysis(index, y_true, y_prob, threshold=0.5)
    assert result["n_false_positive"] == 1
    assert result["n_false_negative"] == 1
    assert result["sample_false_positives"][0]["TransactionID"] == 0
    assert result["sample_false_negatives"][0]["TransactionID"] == 3


def test_error_analysis_rates_are_bounded():
    index = pd.RangeIndex(10)
    result = error_analysis(index, Y_TRUE, Y_PROB_PERFECT, threshold=0.5)
    assert 0.0 <= result["false_positive_rate_of_negatives"] <= 1.0
    assert 0.0 <= result["false_negative_rate_of_positives"] <= 1.0


def test_error_analysis_caps_samples():
    index = pd.RangeIndex(200)
    y_true = np.ones(200, dtype=int)
    y_prob = np.zeros(200)  # every row is a false negative
    result = error_analysis(index, y_true, y_prob, threshold=0.5)
    assert result["n_false_negative"] == 200
    assert len(result["sample_false_negatives"]) == 20  # capped
