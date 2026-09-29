"""Turns the coverage into the sentence somebody has to act on."""


TRUST = "Trust it inside its coverage, and read the refusals."
TOO_QUIET = "It refuses too much to be useful."
NOT_GOOD = "Do not ship it. It is wrong too often on what it does answer."
KEEP_NEW = "Keep the new model."
KEEP_OLD = "Keep the old model."
USEFUL = 0.60
RIGHT_ENOUGH = 0.80


def on_coverage(measured):
    if measured["coverage"] < USEFUL:
        return {
            "verdict": TOO_QUIET,
            "reason": (
                f"It answered {measured['coverage']:.0%} of the pictures. A "
                "program that says nothing four times in ten is a person's "
                "job with extra steps."
            ),
        }
    if measured["accuracy"] < RIGHT_ENOUGH:
        return {
            "verdict": NOT_GOOD,
            "reason": (
                f"{measured['accuracy']:.1%} right on the ones it chose to "
                "answer. Refusing the hard ones did not make it good at the "
                "easy ones."
            ),
        }
    return {
        "verdict": TRUST,
        "reason": (
            f"{measured['accuracy']:.1%} right across "
            f"{measured['coverage']:.0%} of the pictures. The rest it handed "
            "back rather than guessing."
        ),
    }


def on_relearn(compared):
    before, after = compared["before"], compared["after"]
    better = after["accuracy"] >= before["accuracy"]
    wider = after["coverage"] >= before["coverage"]
    if better and wider:
        return {
            "verdict": KEEP_NEW,
            "reason": (
                f"It judges {after['coverage']:.0%} against "
                f"{before['coverage']:.0%}, and is right "
                f"{after['accuracy']:.1%} against {before['accuracy']:.1%}."
            ),
        }
    if better:
        return {
            "verdict": KEEP_NEW,
            "reason": (
                f"Accuracy rose to {after['accuracy']:.1%}, and it answers "
                f"{after['coverage']:.0%} rather than {before['coverage']:.0%}. "
                "Fewer answers, better ones."
            ),
        }
    return {
        "verdict": KEEP_OLD,
        "reason": (
            f"The corrections made it worse: {after['accuracy']:.1%} against "
            f"{before['accuracy']:.1%}. Check who wrote the labels."
        ),
    }
