"""The pictures this repository runs on: handwritten digits.

1,797 eight by eight photographs that ship inside scikit-learn, so the
demonstration needs no download, no key and no network.

The question is "is this a three". Threes are confused with fives,
eights and nines often enough to make the question real.

The strangers are pictures that are not digits at all: a digit with its
pixels shuffled, and plain noise. They are invented on purpose and the
readme says so. Using unseen digits instead would be dishonest, because
a four really does look like the things the model was trained on, and
refusing it would be the wrong behaviour.
"""

import sys

import numpy
from sklearn.datasets import load_digits


TARGET = 3
TOP = 16.0


def load(seed=1):
    digits = load_digits()
    return {
        "pictures": digits.data,
        "is_target": (digits.target == TARGET).astype(int),
        "strangers": make_strangers(digits.data, seed),
        "top": TOP,
        "name": "handwritten digits, is it a three",
    }


def make_strangers(pictures, seed, count=300):
    """Half shuffled digits, half noise. Neither is a handwritten number."""
    source = numpy.random.RandomState(seed)
    chosen = pictures[source.permutation(len(pictures))[: count // 2]]
    shuffled = numpy.array([source.permutation(row) for row in chosen])
    noise = source.randint(0, int(TOP) + 1, size=(count // 2, pictures.shape[1]))
    return numpy.vstack([shuffled, noise]).astype(float)


def main():
    loaded = load()
    print(
        f"{len(loaded['pictures']):,} pictures, "
        f"{int(loaded['is_target'].sum()):,} threes, "
        f"{len(loaded['strangers']):,} strangers"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
