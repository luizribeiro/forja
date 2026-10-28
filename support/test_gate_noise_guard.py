"""Reusable noise guard validates budgets and remembers departures independently of runners."""
import pytest
from gate_noise_guard import NoiseGuard


def test_guard_accepts_comparable_block_noise():
    guard = NoiseGuard({"block_sd": .001})
    report = guard.observe([-.001, .001] * 3, "primary")
    assert not report["collection_fallbacks"]
    assert report["sd_factor"] == 2.
    assert report["stored_block_sd"] == .001


@pytest.mark.parametrize("scale", [.01, .00001])
def test_guard_records_departure_once_with_observed_noise(scale):
    guard = NoiseGuard({"block_sd": .001})
    scores = [-scale, scale] * 3
    for phase in ["primary", "confirmation"]:
        report = guard.observe(scores, phase)
    assert len(report["collection_fallbacks"]) == 1
    event = report["collection_fallbacks"][0]
    assert event["phase"] == "primary" and event["blocks"] == 6
    assert event["observed_sd"] == report["observed_block_sd"]


def test_guard_keeps_drift_alarm_after_later_stability():
    guard = NoiseGuard({"block_sd": .001})
    guard.observe([-.001, .001] * 3, "primary", drift_alarm=True)
    assert guard.observe([-.001, .001] * 4, "primary")["collection_fallbacks"][0]["reason"] == "baseline-drift"


@pytest.mark.parametrize("factor", [0, 1, None, float("inf")])
def test_guard_refuses_invalid_factors(factor):
    with pytest.raises(ValueError, match="factor"):
        NoiseGuard({"history_sd_factor": factor})
