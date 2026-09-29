from cat_check import abstain
from cat_check import coverage_report
from cat_check import model


def trained(training):
    fitted = model.fit(training[0], training[1], top=16)
    return fitted, abstain.learn_the_shape(fitted, training[0])


def test_a_margin_of_zero_judges_everything(training, held_out):
    fitted, shape = trained(training)
    answers = abstain.judge(fitted, shape, held_out[0], margin=0.0)
    measured = coverage_report.measure(answers, held_out[1])
    assert measured["coverage"] > 0.95


def test_a_wider_margin_judges_less_and_is_righter(training, held_out):
    """If this does not hold, the confidence number means nothing."""
    fitted, shape = trained(training)
    narrow = coverage_report.measure(
        abstain.judge(fitted, shape, held_out[0], margin=0.0), held_out[1]
    )
    wide = coverage_report.measure(
        abstain.judge(fitted, shape, held_out[0], margin=0.9), held_out[1]
    )
    assert wide["coverage"] < narrow["coverage"]
    assert wide["accuracy"] >= narrow["accuracy"]


def test_a_stranger_is_refused_as_nothing_like_it(training, strangers):
    fitted, shape = trained(training)
    answers = abstain.judge(fitted, shape, strangers, margin=0.0)
    refused = sum(
        1 for answer in answers
        if not answer["answered"] and answer["reason"] == "nothing like it"
    )
    assert refused / len(answers) > 0.8


def test_ordinary_pictures_are_not_called_strangers(training, held_out):
    fitted, shape = trained(training)
    answers = abstain.judge(fitted, shape, held_out[0], margin=0.0)
    strangers = sum(
        1 for answer in answers
        if not answer["answered"] and answer["reason"] == "nothing like it"
    )
    assert strangers / len(answers) < 0.1


def test_every_refusal_carries_its_reason(training, strangers):
    fitted, shape = trained(training)
    for answer in abstain.judge(fitted, shape, strangers, margin=0.5):
        if not answer["answered"]:
            assert answer["reason"] in ("too unsure", "nothing like it")
