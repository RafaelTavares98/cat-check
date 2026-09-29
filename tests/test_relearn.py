import numpy
import pytest

from cat_check import abstain
from cat_check import model
from cat_check import relearn


def trained(training):
    fitted = model.fit(training[0], training[1], top=16)
    return {"fitted": fitted, "shape": abstain.learn_the_shape(fitted, training[0])}


def test_corrections_are_added_to_the_training_rows(training):
    grown = relearn.fold_in(
        training[0], training[1],
        {"pictures": training[0][:10], "labels": training[1][:10]},
        top=16,
    )
    assert grown["rows"] == len(training[0]) + 10


def test_nothing_to_learn_from_is_refused(training):
    with pytest.raises(ValueError, match="nothing to learn"):
        relearn.fold_in(
            training[0], training[1],
            {"pictures": numpy.empty((0, 64)), "labels": numpy.array([])},
            top=16,
        )


def test_a_picture_without_a_label_is_refused(training):
    with pytest.raises(ValueError, match="the label a person gave it"):
        relearn.fold_in(
            training[0], training[1],
            {"pictures": training[0][:5], "labels": training[1][:2]},
            top=16,
        )


def test_both_models_are_measured_on_the_same_pictures(training, held_out):
    before = trained(training)
    after = relearn.fold_in(
        training[0], training[1],
        {"pictures": held_out[0][:20], "labels": held_out[1][:20]},
        top=16,
    )
    compared = relearn.compare(before, after, held_out[0][20:], held_out[1][20:])
    assert compared["before"]["seen"] == compared["after"]["seen"]
