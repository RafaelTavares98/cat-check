import pytest

from cat_check import coverage_report
from cat_check import report
from cat_check import verdict


def answered(is_cat):
    return {"answered": True, "is_target": is_cat, "chance": 0.9, "distance": 1.0}


def refused(reason="too unsure"):
    return {"answered": False, "reason": reason, "chance": 0.5, "distance": 1.0}


def test_accuracy_counts_only_the_ones_it_judged():
    measured = coverage_report.measure(
        [answered(True), answered(False), refused()], [1, 1, 1]
    )
    assert measured["judged"] == 2
    assert measured["accuracy"] == 0.5
    assert measured["coverage"] == pytest.approx(2 / 3)


def test_the_reasons_are_counted_separately():
    measured = coverage_report.measure(
        [refused("too unsure"), refused("nothing like it"), answered(True)], [1, 1, 1]
    )
    assert measured["refused"] == {"too unsure": 1, "nothing like it": 1}


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="not the same pictures"):
        coverage_report.measure([answered(True)], [1, 0])


def test_a_model_that_refuses_half_is_called_too_quiet():
    measured = {"coverage": 0.5, "accuracy": 0.99, "judged": 50, "seen": 100}
    assert verdict.on_coverage(measured)["verdict"] == verdict.TOO_QUIET


def test_a_model_that_is_often_wrong_is_not_shipped():
    measured = {"coverage": 0.9, "accuracy": 0.7, "judged": 90, "seen": 100}
    assert verdict.on_coverage(measured)["verdict"] == verdict.NOT_GOOD


def test_a_good_one_is_trusted_inside_its_coverage():
    measured = {"coverage": 0.82, "accuracy": 0.91, "judged": 82, "seen": 100}
    assert verdict.on_coverage(measured)["verdict"] == verdict.TRUST


def test_the_report_never_prints_an_accuracy_without_a_coverage():
    measured = {
        "seen": 100, "judged": 82, "coverage": 0.82, "accuracy": 0.913,
        "refused": {"too unsure": 12, "nothing like it": 6},
    }
    printed = report.render_judging(measured, verdict.on_coverage(measured), 10000, "digits")
    for line in printed.split("\n"):
        if "Accuracy" in line:
            assert "of the ones it judged" in line


def test_corrections_that_make_it_worse_keep_the_old_model():
    compared = {
        "before": {"accuracy": 0.90, "coverage": 0.80},
        "after": {"accuracy": 0.80, "coverage": 0.85},
    }
    assert verdict.on_relearn(compared)["verdict"] == verdict.KEEP_OLD
