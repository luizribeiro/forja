"""Unverified marginal or stale noise must never supply a pass-drift budget."""

import hashlib
import json
import statistics

import pytest

from gate_history import history, noise_identity
from gate_stats import METHOD


def identity(workload="same"):
    return {"row": "row", "workload_sha256": hashlib.sha256(workload.encode()).hexdigest(),
            "harness_protocol": {"version": "test-v1", "measurement_protocol": {"warmups": 1}},
            "machine_identity": {"hostname": "test", "device": "gpu", "description": "gpu"},
            "os_build": "test-os", "guard": []}


def history_row(tmp_path, scores=None, created=100, workload="same"):
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"identity": identity(workload), "method": METHOD, "created_at": created,
                               "log_scores": scores if scores is not None else [-.003, .001, -.002, .002, .001, .003]}))
    return {"id": "row", "history_identity": identity(), "block_sd": 99, "D": 99,
            "history": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "max_age_seconds": 10}}


def test_matching_raw_history_supplies_only_its_observed_block_noise(tmp_path):
    row = history_row(tmp_path)
    result = history(row, now=105)
    expected = statistics.stdev([-.003, .001, -.002, .002, .001, .003])
    assert result["D"] == result["block_sd"] == expected
    assert result["history_status"] == "verified"


@pytest.mark.parametrize("scores,created,now,status", [
    ([0, 1, 2, 3, 4], 100, 105, "insufficient"),
    ([0] * 6, 100, 105, "zero-variance"),
    ([0, 1, 2, 3, 4, 5], 100, 110, "stale-or-undated"),
    ([0, 1, 2, 3, 4, 5], 106, 105, "stale-or-undated"),
])
def test_unusable_history_does_not_supply_a_budget(tmp_path, scores, created, now, status):
    result = history(history_row(tmp_path, scores, created), now=now)
    assert result["history_status"] == status
    assert "D" not in result and "block_sd" not in result


def test_mismatched_or_unverified_history_is_refused(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        history(history_row(tmp_path, workload="other"), now=105)
    row = history_row(tmp_path)
    row["history"]["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="artifact"):
        history(row, now=105)
    del row["history"]["sha256"]
    with pytest.raises(ValueError, match="pinned"):
        history(row, now=105)

def test_no_history_never_trusts_a_declared_marginal_noise_budget():
    row = history({"id": "row", "block_sd": .001, "D": .001})
    assert "block_sd" not in row and "D" not in row


def test_build_changes_do_not_invalidate_workload_noise(tmp_path):
    row = history_row(tmp_path)
    row.update(baseline_source="new-host", candidate_source="new-engine",
               baseline={"binary_sha256": "new", "component_sha256": "new"})
    assert history(row, now=105)["history_status"] == "verified"
    assert set(history(row, now=105)["history_identity"]) == {
        "row", "workload_sha256", "harness_protocol", "machine_identity", "os_build", "guard"}


@pytest.mark.parametrize("field,value", [("workload_sha256", "0" * 64),
    ("harness_protocol", {"version": "changed", "measurement_protocol": {"warmups": 1}}),
    ("os_build", "changed"), ("guard", ["wrapper", "128"])])
def test_noise_identity_rejects_workload_protocol_or_os_changes(tmp_path, field, value):
    row = history_row(tmp_path)
    row["history_identity"][field] = value
    with pytest.raises(ValueError, match="identity"):
        history(row, now=105)


def test_legacy_history_needs_an_explicit_audited_workload_hash(tmp_path):
    row = history_row(tmp_path)
    legacy = {"row": "row", "protocol": "test-v1", "measurement_protocol": {"warmups": 1},
              "machine_identity": {"hostname": "test", "device": "gpu", "os_build": "test-os"},
              "machine": "gpu", "candidate_binary_sha256": "old", "baseline_source": "old"}
    assert noise_identity(legacy, row["history_identity"]["workload_sha256"], []) == identity()
    with pytest.raises(ValueError, match="audited"):
        noise_identity(legacy)
