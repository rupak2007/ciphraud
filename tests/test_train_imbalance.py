import numpy as np
import pytest

from src.train.imbalance import class_weight_dict, compute_scale_pos_weight


def test_compute_scale_pos_weight_matches_ratio():
    y = np.array([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_compute_scale_pos_weight_balanced_data_is_one():
    y = np.array([0] * 50 + [1] * 50)
    assert compute_scale_pos_weight(y) == pytest.approx(1.0)


def test_compute_scale_pos_weight_raises_on_no_positives():
    y = np.zeros(10)
    with pytest.raises(ValueError, match="no positive examples"):
        compute_scale_pos_weight(y)


def test_compute_scale_pos_weight_uses_only_given_array():
    """The imbalance ratio must come from exactly the array passed in --
    verifies no hidden global state or caching across calls."""
    y_train = np.array([0] * 97, dtype=int)
    y_train = np.append(y_train, [1, 1, 1])  # 3/97 imbalance
    ratio_a = compute_scale_pos_weight(y_train)
    y_other = np.array([0] * 50 + [1] * 50)  # a completely different, balanced array
    ratio_b = compute_scale_pos_weight(y_other)
    assert ratio_a != pytest.approx(ratio_b)
    assert ratio_a == pytest.approx(97 / 3)


def test_class_weight_dict_matches_sklearn_balanced_formula():
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([0] * 80 + [1] * 20)
    ours = class_weight_dict(y)
    sklearn_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    assert ours[0] == pytest.approx(sklearn_weights[0])
    assert ours[1] == pytest.approx(sklearn_weights[1])


def test_class_weight_dict_minority_class_gets_higher_weight():
    y = np.array([0] * 95 + [1] * 5)
    weights = class_weight_dict(y)
    assert weights[1] > weights[0]


def test_class_weight_dict_balanced_data_gives_equal_weights():
    y = np.array([0] * 50 + [1] * 50)
    weights = class_weight_dict(y)
    assert weights[0] == pytest.approx(weights[1])


def test_class_weight_dict_raises_on_missing_class():
    y = np.ones(10)  # no class-0 examples
    with pytest.raises(ValueError, match="no examples of class 0"):
        class_weight_dict(y)
