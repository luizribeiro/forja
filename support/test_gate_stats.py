"""Boundary coverage, realistic noise and periodic bias tests for gate decisions."""

from fractions import Fraction
import math
import random

import pytest

from gate_stats import RANKS, block_score, decide, drift_promotes, log_tau, operating_characteristics


def test_exact_boundary_spends_the_calibrated_budget():
    result = operating_characteristics()
    assert result["pass"] == result["fail"] == Fraction(26843154, 1073741824)
    assert .02499 < result["pass"] < Fraction(1, 40)
    assert sum(result[key] for key in ["pass", "fail", "inconclusive"]) == 1


def test_thresholds_ties_cap_and_drift_cannot_rescue_failure():
    assert decide([0.] * 6, "non-regression")["decision"] == "pass"
    assert decide([0.] * 6, "equivalence")["decision"] == "pass"
    assert decide([0.] * 6, "equivalence")["diagnostic"]
    assert log_tau("non-regression", .02, 0) == math.log(.98)
    assert log_tau("improvement", .02, .03) == math.log(1.03)
    assert decide([math.log(.98)] * 30, "non-regression")["decision"] == "inconclusive"
    assert decide([-.1] * 6, "non-regression", drift_alarm=True)["decision"] == "fail"
    assert decide([0.] * 6, "non-regression", drift_alarm=True)["decision"] == "invalidated"
    assert decide([0.] * 30, "improvement")["decision"] == "inconclusive"
    assert drift_promotes([0.] * 5, None)
    assert drift_promotes([-.02] * 5, .01)
    assert not drift_promotes([0.] * 5, .01)


@pytest.mark.parametrize("order", ["ABBA", "BAAB"])
def test_midpoint_interpolation_cancels_unequal_linear_drift(order):
    windows = [(0, 1), (2, 4), (5, 8), (9, 13)]
    samples = [{"side": side, "start": start, "end": end,
                "score": math.exp(.01 * (start + end) / 2 + (.03 if side == "B" else 0))}
               for side, (start, end) in zip(order, windows)]
    assert block_score(samples) == pytest.approx(.03)
    swapped = [{**s, "side": "B" if s["side"] == "A" else "A"} for s in samples]
    assert block_score(swapped) == pytest.approx(-block_score(samples))
    samples[1]["start"] = 0
    with pytest.raises(ValueError):
        block_score(samples)


@pytest.mark.parametrize("sd", [
    .01159785, .00478149, .01466125, .00495862, .01574778, .00448250,
    .01094970, .00410829, .01196321, .00375588, .01085187, .00367644,
    .00637128, .02239088, .00692087, .00341831, .00708620, .00292411,
    .01286470, .00290327, .01145032, .00263269, .00911471, .01640612,
])
def test_power_at_every_historical_conversation_block_noise(sd):
    rng = random.Random(913)
    passed = 0
    for _ in range(400):
        scores = []
        for n in range(1, 31):
            scores.append(rng.gauss(0, sd))
            if n in RANKS:
                decision = decide(scores, "non-regression")["decision"]
                if decision != "continue":
                    passed += decision == "pass"
                    break
    expected = operating_characteristics(
        .5 * (1 + math.erf(-math.log(.98) / sd / math.sqrt(2))))["pass"]
    assert abs(passed / 400 - expected) < .07


@pytest.mark.parametrize("noise", ["gaussian", "heavy-tail", "spikes", "periodic-ar"])
def test_randomized_orientation_controls_boundary_calls(noise):
    rng = random.Random(731)
    calls = 0
    trials = 3000
    for _ in range(trials):
        scores = []
        background = 0.
        for n in range(1, 31):
            order = "ABBA" if rng.getrandbits(1) else "BAAB"
            samples = []
            for position, side in enumerate(order):
                background = .7 * background + rng.gauss(0, .01)
                value = {"gaussian": rng.gauss(0, .02),
                         "heavy-tail": .01 * rng.gauss(0, 1) / max(abs(rng.gauss(0, 1)), .01),
                         "spikes": .1 if rng.random() < .05 else 0.,
                         "periodic-ar": background + (.08 if position in [0, 3] else 0.)}[noise]
                samples.append({"side": side, "start": position * 2.,
                                "end": position * 2. + 1, "score": math.exp(value)})
            scores.append(block_score(samples))
            if n in RANKS and decide(scores, "improvement")["decision"] != "continue":
                calls += decide(scores, "improvement")["decision"] in {"pass", "fail"}
                break
    assert calls / trials < .065
