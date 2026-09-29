"""The only file that reads the command line."""

import argparse
import sys

from cat_check import report
from cat_check import run
from data import digits


def main(argv=None):
    options = _read_options(argv)
    parts = run.split(digits.load())
    trained = run.train(parts)
    if options.command == "judge":
        return _judge(parts, trained, options)
    if options.command == "curve":
        return _curve(parts, trained)
    return _learn(parts, trained, options)


def _judge(parts, trained, options):
    judged = run.judge(trained, parts["test_x"], parts["test_y"], options.margin)
    print(report.render_judging(judged["measured"], judged["verdict"], trained["rows"], parts["name"]))
    refused = run.refusal_rate(trained, parts["stranger_x"], options.margin)
    print(
        f"Strangers ............. {refused:.0%} of "
        f"{len(parts['stranger_x']):,} pictures that are not digits at all were refused"
    )
    return 0 if judged["verdict"]["verdict"].startswith("Trust") else 1


def _curve(parts, trained):
    print(f"{'margin':>8}  {'coverage':>9}  {'accuracy':>9}")
    for margin in (0.0, 0.2, 0.4, 0.6):
        measured = run.judge(
            trained, parts["test_x"], parts["test_y"], margin
        )["measured"]
        print(
            f"{margin:>8.1f}  {measured['coverage']:>8.0%}  "
            f"{measured['accuracy']:>8.1%}"
        )
    return 0


def _learn(parts, trained, options):
    learned = run.learn_from_mistakes(parts, trained, options.margin)
    print(
        report.render_relearn(
            learned["compared"], learned["corrections"], learned["verdict"]
        )
    )
    return 0


def _read_options(argv):
    parser = argparse.ArgumentParser(
        prog="cat-check",
        description="Answers about a picture, or says nothing and explains why.",
    )
    parser.add_argument("command", choices=["judge", "curve", "learn"])
    parser.add_argument("--margin", type=float, default=0.15)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
