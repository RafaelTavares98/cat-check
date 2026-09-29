import numpy
import pytest


TOP = 16.0


def make_pictures(count, kind, seed):
    """Invented 8x8 pictures, so the suite needs no download at all.

    The two classes overlap on purpose. A pair that separates perfectly
    would make every probability a 0 or a 1, and then the abstention
    tests would prove nothing.
    """
    source = numpy.random.RandomState(seed)
    rows = source.randint(4, 9, size=(count, 64)).astype(float)
    if kind == "target":
        rows[:, :16] += source.normal(2.0, 1.5, size=(count, 1))
    elif kind == "other":
        rows[:, 48:] += source.normal(2.0, 1.5, size=(count, 1))
    else:
        rows = source.randint(13, 17, size=(count, 64)).astype(float)
    return numpy.clip(rows, 0, TOP)


@pytest.fixture
def training():
    pictures = numpy.vstack(
        [make_pictures(300, "target", seed=1), make_pictures(300, "other", seed=2)]
    )
    return pictures, numpy.array([1] * 300 + [0] * 300)


@pytest.fixture
def held_out():
    pictures = numpy.vstack(
        [make_pictures(100, "target", seed=3), make_pictures(100, "other", seed=4)]
    )
    return pictures, numpy.array([1] * 100 + [0] * 100)


@pytest.fixture
def strangers():
    return make_pictures(80, "stranger", seed=5)
