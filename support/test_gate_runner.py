"""A failed row gets one confirmation; a failed improvement never gets a rescue."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tomllib

from gate_runner import run_row


def sampler(scores, calls):
    clock = 0.

    def sample(side, reps, phase, block, leg):
        nonlocal clock
        clock += 2
        calls.append((phase, side, reps))
        return {"side": side, "start": clock, "end": clock + 1,
                "score": scores.get(phase, 1.) if side == "B" else 1.,
                "record": {"schema_version": 5, "config": {"bench": {"reps": reps}},
                           "results": [{"pp": {}, "tg": {}}]}}
    return sample


def test_one_fresh_confirmation_records_both_without_rescuing_a_failure():
    row = {"id": "row", "profile": "profile", "D": .003}
    calls = []
    result = run_row(row, sampler({"primary": .9, "confirmation": 1.}, calls), 42)
    assert result["decision"] == "unconfirmed-fail"
    assert len(result["runs"]) == 2
    assert len(calls) == 48
    assert not result["confirmed_regression"]
    confirmed = run_row(row, sampler({"primary": .9, "confirmation": .9}, []), 42)
    assert confirmed["confirmed_regression"]
    row["question"] = "improvement"
    failed = run_row(row, sampler({"primary": .9}, []), 42)
    assert len(failed["runs"]) == 1
    assert not failed["improvement_claim"]


def test_classic_mode_preserves_native_record_shape_and_exact_rep_counts():
    calls = []
    reference_path = sorted(Path(__file__).resolve().parents[1].glob("engines/*/bench/*/accepted.json"))[0]
    reference = json.loads(reference_path.read_text())
    native_sample = sampler({}, calls)

    def fixed_sample(*arguments):
        packet = native_sample(*arguments)
        packet["record"] = deepcopy(reference)
        return packet

    result = run_row({"id": "row", "profile": "profile"}, fixed_sample, 7,
                     no_early_stop=True, reps=30)
    assert [side for _, side, _ in calls] == list("ABBA")
    assert [reps for _, _, reps in calls] == [30] * 4
    assert not result["runs"]
    for record in result["records"]:
        assert record["record"] == reference
        assert tomllib.loads(record["record"]["config"])["bench"]["reps"] == 30


def test_drift_promotion_keeps_screen_separate_from_sequential_samples():
    row = {"id": "row", "profile": "profile", "D": .01}
    result = run_row(row, sampler({"drift": .95}, []), 17, affected=False)
    assert len(result["drift_blocks"]) == 5
    assert result["promoted"]
    assert len(result["runs"][0]["blocks"]) == 6
    result = run_row(row, sampler({}, []), 17, affected=False)
    assert result["decision"] == "unchanged-drift-check"
    row["question"] = "equivalence"
    assert run_row(row, sampler({}, []), 17)["decision"] == "no change"


def test_baseline_drift_uses_verified_noise_budget_instead_of_regression_tolerance():
    def run(total_drift):
        clock = 0.

        def sample(side, _reps, _phase, block, _leg):
            nonlocal clock
            clock += 2
            return {"side": side, "start": clock, "end": clock + 1,
                    "score": math.exp(total_drift * block / 5)}
        return run_row({"id": "row", "profile": "profile", "D": .003, "X": .02}, sample, 7)

    above = run(.01)
    assert above["decision"] == "invalidated"
    assert above["runs"][0]["pass_drift_budget"] == .003
    assert len(above["runs"][0]["blocks"]) == 6
    assert run(.002)["decision"] == "pass"


def test_no_verified_noise_history_cannot_claim_a_protected_gating_pass():
    result = run_row({"id": "row", "profile": "profile"}, sampler({}, []), 7)
    assert result["decision"] == "history-collected"
    assert result["drift_protection"] == "unavailable"
    assert len(result["runs"][0]["blocks"]) == 30
