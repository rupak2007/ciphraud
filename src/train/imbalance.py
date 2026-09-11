"""Class-imbalance handling for the Phase 3 baselines.

docs/eda.md Sec.1 measured a 3.499% global fraud rate. CLAUDE.md Sec.6
requires "class-imbalance handling appropriate to a ~3.5% fraud rate" and
`docs/plan.md`'s Phase 3 fallback strategy lists, in order: "resampling,
class weights, focal-loss-style objectives." This project uses **class
weighting** -- the first-listed, simplest option -- for two concrete
reasons, not just because it's listed first:

1. It needs no new dependency (SMOTE-style resampling would require
   `imbalanced-learn`, not currently a project dependency and not
   justified by any other requirement -- CLAUDE.md Sec.13/Sec.15).
2. It is natively supported by both required model types (scikit-learn's
   `class_weight`, XGBoost's `scale_pos_weight`), so no custom resampling
   pipeline is needed that would also have to be kept leakage-safe (a
   resampler fit on train, like any other statistic, would need the same
   train-only-fit discipline as `src/data/preprocess.py`'s encoders).

The weight/ratio is always computed from **TRAIN ONLY** -- computing it
over train+val (or worse, the full dataset) would leak validation-period
class-balance information into training, exactly the kind of statistic
CLAUDE.md Sec.6 requires to be past-scoped.
"""

from __future__ import annotations

import numpy as np


def compute_scale_pos_weight(y_train) -> float:
    """XGBoost's `scale_pos_weight`: n_negative / n_positive, from train only.

    This is the standard closed-form weighting XGBoost's own documentation
    recommends for imbalanced binary classification -- it rebalances the
    gradient contribution of the minority (fraud) class to be equal in
    aggregate to the majority class, without duplicating or discarding any
    row.
    """
    y_train = np.asarray(y_train)
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    if n_positive == 0:
        raise ValueError("compute_scale_pos_weight: no positive examples in y_train")
    return n_negative / n_positive


def class_weight_dict(y_train) -> dict[int, float]:
    """scikit-learn-compatible class_weight, equivalent to class_weight='balanced'
    but computed explicitly (and logged) from train only, per this module's
    docstring -- rather than relying on sklearn's internal 'balanced' string
    to implicitly do the right scoping (it does, since it's computed from
    whatever y is passed to .fit(), but making the number explicit here
    means it's visible in results/phase3_baselines/ provenance rather than
    only implicit in a fitted estimator's internals).

    weight[c] = n_samples / (n_classes * n_samples_of_class_c) -- scikit-learn's
    exact 'balanced' formula (sklearn.utils.class_weight.compute_class_weight).
    """
    y_train = np.asarray(y_train)
    n_samples = len(y_train)
    n_classes = 2
    weights = {}
    for c in (0, 1):
        n_c = int((y_train == c).sum())
        if n_c == 0:
            raise ValueError(f"class_weight_dict: no examples of class {c} in y_train")
        weights[c] = n_samples / (n_classes * n_c)
    return weights
