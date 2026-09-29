"""Prints the two blocks, and never an accuracy without its coverage."""


REASONS = {
    "too unsure": "the two classes were within the margin",
    "nothing like it": "further from the training pictures than the limit",
}


def render_judging(measured, verdict, trained_on, name="pictures"):
    lines = [
        _row("Model", f"logistic on {trained_on:,} pictures, {name}"),
        _row(
            "Judged",
            f"{measured['judged']:,} of {measured['seen']:,} pictures, "
            f"{measured['coverage']:.0%} coverage",
        ),
        _row("Accuracy", f"{measured['accuracy']:.1%} of the ones it judged"),
        _row("Refused", f"{measured['seen'] - measured['judged']:,} pictures"),
    ]
    for reason, count in sorted(measured["refused"].items()):
        lines.append(f"  {reason} {'.' * max(1, 20 - len(reason))} {count:,}, {REASONS[reason]}")
    return "\n".join(lines + _closing(verdict))


def render_relearn(compared, corrections, verdict):
    before, after = compared["before"], compared["after"]
    lines = [
        _row("Corrections", f"{corrections:,} pictures a person relabelled"),
        _row("Before", f"{before['accuracy']:.1%} at {before['coverage']:.0%} coverage"),
        _row("After", f"{after['accuracy']:.1%} at {after['coverage']:.0%} coverage"),
    ]
    return "\n".join(lines + _closing(verdict))


def _closing(verdict):
    return [_row("Verdict", verdict["verdict"]), _row("Reason", verdict["reason"])]


def _row(label, value):
    return f"{label} {'.' * max(1, 22 - len(label))} {value}"
