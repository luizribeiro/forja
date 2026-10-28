"""Observed block noise and drift can disable reuse, never erase a failure."""
import math
import pytest
from gate_runner import run_row


def sample_noise(scale, offset=0., recover=False):
    def sample(side, _reps, phase, block, leg):
        effect = .1 if recover and (phase == "confirmation" or block >= 6) else offset
        log_score = effect + scale * (-1 if block % 2 else 1) if side == "B" else 0.
        return {"side": side, "start": 8 * block + 2 * leg, "end": 8 * block + 2 * leg + 1,
                "score": math.exp(log_score)}
    return sample


def row(factor=2.):
    return {"id": "row", "profile": "profile", "D": .001, "block_sd": .001,
            "history_sd_factor": factor}


@pytest.mark.parametrize("scale", [.01, .00001])
def test_sd_departure_collects_all_blocks_without_accepting(scale):
    result = run_row(row(), sample_noise(scale), 5)
    assert result["decision"] == "history-collected"
    assert not result["improvement_claim"]
    assert len(result["runs"][0]["blocks"]) == 30
    event = result["noise_guard"]["collection_fallbacks"][0]
    assert event["reason"] == "block-sd-departure" and event["blocks"] == 6
    assert event["factor"] == 2.


def test_declared_factor_and_matching_noise_allow_early_stop():
    assert run_row(row(), sample_noise(.001), 5)["decision"] == "pass"
    result = run_row(row(20), sample_noise(.01), 5)
    assert result["decision"] == "pass" and len(result["runs"][0]["blocks"]) == 6
    assert not result["noise_guard"]["collection_fallbacks"]


def test_sd_departure_in_five_block_drift_screen_promotes_to_full_collection():
    result = run_row(row(), sample_noise(.01), 5, affected=False)
    assert result["promoted"] and len(result["drift_blocks"]) == 5
    assert len(result["runs"][0]["blocks"]) == 30


def test_collection_cannot_erase_first_failure_or_add_rescue_runs():
    result = run_row(row(), sample_noise(.01, -.1, recover=True), 5)
    assert result["decision"] == "unconfirmed-fail"
    assert len(result["runs"]) == 2
    assert all(len(run["blocks"]) == 30 for run in result["runs"])
    assert result["runs"][0]["stopping"]["decision"] == "fail"
    assert result["runs"][0]["stopping"]["blocks"] == 6
    confirmed = run_row(row(), sample_noise(.01, -.1), 5)
    assert confirmed["confirmed_regression"]


@pytest.mark.parametrize("factor", [1, 0, float("inf"), float("nan")])
def test_invalid_sd_factor_is_refused(factor):
    with pytest.raises(ValueError, match="factor"):
        run_row(row(factor), sample_noise(.001), 5)
