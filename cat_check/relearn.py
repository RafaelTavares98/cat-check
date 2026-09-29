"""Folds what a person corrected back into the training set.

The pictures worth learning from are the ones the model got wrong and
the ones it refused. Those are exactly the pictures a normal training
set does not contain, which is why a model that never collects them
stops improving on the day it ships.
"""

import numpy

from cat_check import abstain
from cat_check import coverage_report
from cat_check import model


def fold_in(train_pictures, train_labels, corrections, top=255.0, seed=1):
    """A second model, trained on the old rows plus the corrected ones."""
    _demand_corrections(corrections)
    pictures = numpy.vstack([train_pictures, corrections["pictures"]])
    labels = numpy.concatenate([train_labels, corrections["labels"]])
    fitted = model.fit(pictures, labels, top, seed=seed)
    return {
        "fitted": fitted,
        "shape": abstain.learn_the_shape(fitted, pictures),
        "rows": len(pictures),
    }


def compare(before, after, pictures, truth, margin=0.15):
    """Both models on the same held-out pictures, or it proves nothing."""
    return {
        "before": coverage_report.measure(
            abstain.judge(before["fitted"], before["shape"], pictures, margin), truth
        ),
        "after": coverage_report.measure(
            abstain.judge(after["fitted"], after["shape"], pictures, margin), truth
        ),
    }


def _demand_corrections(corrections):
    if len(corrections["pictures"]) == 0:
        raise ValueError("nobody corrected anything, so there is nothing to learn")
    if len(corrections["pictures"]) != len(corrections["labels"]):
        raise ValueError("every corrected picture needs the label a person gave it")
