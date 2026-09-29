"""Trains, judges and relearns, on whichever pictures the loader gives.

The strangers are the honest part: pictures from classes the model was
never shown. Any classifier will answer about them. This one has to
refuse them.
"""

import numpy

from cat_check import abstain
from cat_check import coverage_report
from cat_check import model
from cat_check import relearn
from cat_check import verdict


def split(loaded, train_share=0.7, seed=1):
    source = numpy.random.RandomState(seed)
    order = source.permutation(len(loaded["pictures"]))
    cut = int(len(order) * train_share)
    train, test = order[:cut], order[cut:]
    return {
        "train_x": loaded["pictures"][train],
        "train_y": loaded["is_target"][train],
        "test_x": loaded["pictures"][test],
        "test_y": loaded["is_target"][test],
        "stranger_x": loaded["strangers"],
        "top": loaded["top"],
        "name": loaded["name"],
    }


def train(parts, seed=1):
    fitted = model.fit(parts["train_x"], parts["train_y"], parts["top"], seed=seed)
    return {
        "fitted": fitted,
        "shape": abstain.learn_the_shape(fitted, parts["train_x"]),
        "rows": len(parts["train_x"]),
    }


def judge(trained, pictures, truth, margin=0.15):
    answers = abstain.judge(trained["fitted"], trained["shape"], pictures, margin)
    measured = coverage_report.measure(answers, truth)
    return {
        "answers": answers,
        "measured": measured,
        "verdict": verdict.on_coverage(measured),
    }


def refusal_rate(trained, pictures, margin=0.15):
    """The share of a set this model declines to judge at all."""
    answers = abstain.judge(trained["fitted"], trained["shape"], pictures, margin)
    return sum(1 for answer in answers if not answer["answered"]) / len(answers)


def coverage_curve(trained, pictures, truth, margins=(0.0, 0.2, 0.4, 0.6)):
    """Accuracy at several coverages, which is the whole claim in a table."""
    return [
        judge(trained, pictures, truth, margin)["measured"] for margin in margins
    ]


def learn_from_mistakes(parts, trained, margin=0.15, seed=1):
    """Corrects what it got wrong and what it refused, then measures again."""
    answers = abstain.judge(
        trained["fitted"], trained["shape"], parts["test_x"], margin
    )
    hard = [
        place
        for place, answer in enumerate(answers)
        if not answer["answered"]
        or answer["is_target"] != bool(parts["test_y"][place])
    ]
    half = len(hard) // 2
    taught, held = hard[:half], hard[half:]
    _demand_something_to_hold_back(taught, held)
    after = relearn.fold_in(
        parts["train_x"],
        parts["train_y"],
        {"pictures": parts["test_x"][taught], "labels": parts["test_y"][taught]},
        top=parts["top"],
        seed=seed,
    )
    ordinary = [n for n in range(len(parts["test_y"])) if n not in set(hard)]
    check = held + ordinary
    compared = relearn.compare(
        trained, after, parts["test_x"][check], parts["test_y"][check], margin
    )
    return {
        "corrections": len(taught),
        "compared": compared,
        "verdict": verdict.on_relearn(compared),
    }


def _demand_something_to_hold_back(taught, held):
    if not taught or not held:
        raise ValueError(
            "there were too few mistakes to both learn from and check "
            "against. Judge a larger set."
        )
