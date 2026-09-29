import numpy
import pytest

from cat_check import features
from cat_check import model


def test_pixels_come_back_between_zero_and_one(training):
    built = features.build(training[0], top=16)
    assert built.min() >= 0.0 and built.max() <= 1.0


def test_a_picture_of_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match="brighter"):
        features.build(numpy.full((2, 100), 999.0))


def test_a_stack_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="dimensions"):
        features.build(numpy.zeros((2, 32, 32, 3)))


def test_it_beats_a_coin_flip_on_pictures_it_never_saw(training, held_out):
    fitted = model.fit(training[0], training[1], top=16)
    chances = model.chance_of_target(fitted, held_out[0])
    right = ((chances >= 0.5).astype(int) == held_out[1]).mean()
    assert right > 0.7


def test_the_answer_is_a_probability(training, held_out):
    chances = model.chance_of_target(model.fit(training[0], training[1], top=16), held_out[0])
    assert chances.min() >= 0.0 and chances.max() <= 1.0
