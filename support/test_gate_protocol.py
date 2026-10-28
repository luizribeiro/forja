"""Noise histories must match the effective benchmark protocol."""
import pytest


def test_history_identity_tracks_effective_execution_and_priming_modes():
    from gate_worker import bind_measurement_protocol, measurement_protocol
    adapter = {"command": ["/forja", "bench"], "worker_protocol": 2,
               "cache_state": "prefix-fork-per-worker", "expected_provenance": {
                   "/provenance/binary_sha256": "binary", "/inputs/0/engine_sha256": "guest"}}
    row = {"baseline": adapter, "candidate": dict(adapter), "history_identity": {"row": "example"}}
    resident = bind_measurement_protocol(row, True, 1)
    classic = bind_measurement_protocol(row, False, 1)
    assert resident["history_identity"] != classic["history_identity"]
    assert resident["history_identity"]["measurement_protocol"]["worker_protocol"] == 2
    assert classic["history_identity"]["measurement_protocol"]["priming"] == "legacy"
    assert bind_measurement_protocol(row, True, 3)["history_identity"] != resident["history_identity"]
    with pytest.raises(ValueError, match="history measurement"):
        bind_measurement_protocol({**resident, "history": {"path": "pinned-history"}}, False, 1)
    assert bind_measurement_protocol({**resident, "history": {"path": "pinned-history"}}, True, 1)
    http = {"command": ["python", "serving_bench.py", "--priming", "bounded"]}
    assert measurement_protocol({"baseline": http, "candidate": http}, True, 1)["priming"] == "bounded"
    with pytest.raises(ValueError, match="priming protocols"):
        measurement_protocol({"baseline": http, "candidate": {"command": ["python", "legacy.py"]}}, True, 1)


def test_canonical_noise_identity_binds_effective_protocol_inside_version():
    from gate_worker import bind_measurement_protocol
    from test_gate_history_loading import identity
    adapter = {"command": ["/forja", "bench"], "worker_protocol": 2,
               "cache_state": "prefix-fork-per-worker", "expected_provenance": {
                   "/provenance/binary_sha256": "binary", "/inputs/0/engine_sha256": "guest"}}
    row = {"baseline": adapter, "candidate": adapter, "history_identity": identity()}
    bound = bind_measurement_protocol(row, True, 1)["history_identity"]
    assert "measurement_protocol" not in bound
    assert bound["harness_protocol"]["measurement_protocol"]["worker_protocol"] == 2


def test_nested_protocol_missing_measurement_has_controlled_behavior():
    from gate_worker import bind_measurement_protocol
    adapter = {"command": ["python", "serving_bench.py", "--priming", "bounded"]}
    row = {"baseline": adapter, "candidate": adapter,
           "history_identity": {"harness_protocol": {"version": "v2"}}}
    assert bind_measurement_protocol(row, True, 1)["history_identity"]["harness_protocol"]["measurement_protocol"]
    with pytest.raises(ValueError, match="history measurement"):
        bind_measurement_protocol({**row, "history": {"path": "pinned"}}, True, 1)
