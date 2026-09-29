"""The two ways this program says nothing.

**Unsure.** The two classes sit within a hair of each other. Answering
is a coin flip wearing a percentage.

**Nothing like it.** The picture is further from everything the model
was trained on than almost all of that training data is. A classifier
asked about a lorry will answer cat or not-cat, because those are the
only two words it has. This measures the distance and declines.
"""

import numpy

from cat_check import model


FAR = 99.0


def learn_the_shape(fitted, pictures, far=FAR):
    """How far a picture is allowed to be before it counts as a stranger."""
    small = model.small(fitted, pictures)
    middle = small.mean(axis=0)
    spread = small.std(axis=0)
    spread[spread == 0] = 1.0
    distances = _distance(small, middle, spread)
    return {
        "middle": middle,
        "spread": spread,
        "limit": float(numpy.percentile(distances, far)),
    }


def judge(fitted, shape, pictures, margin=0.15):
    """One answer per picture: cat, not a cat, or a refusal with a reason."""
    chances = model.chance_of_target(fitted, pictures)
    distances = _distance(
        model.small(fitted, pictures), shape["middle"], shape["spread"]
    )
    answers = []
    for chance, distance in zip(chances, distances):
        answers.append(_one(chance, distance, shape["limit"], margin))
    return answers


def _one(chance, distance, limit, margin):
    if distance > limit:
        return {
            "answered": False,
            "reason": "nothing like it",
            "chance": float(chance),
            "distance": float(distance),
        }
    if abs(chance - 0.5) < margin / 2:
        return {
            "answered": False,
            "reason": "too unsure",
            "chance": float(chance),
            "distance": float(distance),
        }
    return {
        "answered": True,
        "is_target": bool(chance >= 0.5),
        "chance": float(chance),
        "distance": float(distance),
    }


def _distance(small, middle, spread):
    return numpy.sqrt((((small - middle) / spread) ** 2).sum(axis=1))
