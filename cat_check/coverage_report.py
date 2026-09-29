"""Accuracy at a coverage, because accuracy alone is a half sentence.

A model that answers everything at 78% and a model that answers four
fifths at 91% are different products. One number cannot tell them
apart.
"""


def measure(answers, truth):
    _demand_same_length(answers, truth)
    judged = [
        (answer, real) for answer, real in zip(answers, truth) if answer["answered"]
    ]
    right = sum(1 for answer, real in judged if answer["is_target"] == bool(real))
    return {
        "seen": len(answers),
        "judged": len(judged),
        "coverage": len(judged) / len(answers),
        "accuracy": right / len(judged) if judged else 0.0,
        "refused": _why_refused(answers),
    }


def _why_refused(answers):
    counted = {}
    for answer in answers:
        if not answer["answered"]:
            counted[answer["reason"]] = counted.get(answer["reason"], 0) + 1
    return counted


def _demand_same_length(answers, truth):
    if len(answers) != len(truth):
        raise ValueError(
            f"{len(answers)} answers against {len(truth)} real labels, so "
            "they are not the same pictures"
        )
    if not answers:
        raise ValueError("no pictures were judged, so there is nothing to score")
