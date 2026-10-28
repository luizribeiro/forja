"""Pinned legacy noise remains reusable when only measured build artifacts change."""
import hashlib
import json

import pytest
from gate_history import history
from test_gate_history_loading import history_row, identity


def test_legacy_noise_survives_host_and_engine_changes(tmp_path):
    row = history_row(tmp_path)
    path = tmp_path / "history.json"
    record = json.loads(path.read_text())
    current = identity()
    record["identity"] = {"row": "row", "protocol": "test-v1", "measurement_protocol": {"warmups": 1},
        "machine_identity": {"hostname": "test", "device": "gpu", "os_build": "test-os"},
        "machine": "gpu", "candidate_source": "old", "candidate_binary_sha256": "old",
        "candidate_component_sha256": "old"}
    path.write_text(json.dumps(record))
    pinned = hashlib.sha256(path.read_bytes()).hexdigest()
    row["history"].update(sha256=pinned, legacy_workload_sha256=current["workload_sha256"], legacy_guard=[])
    for build in ["old", "new"]:
        host, engine = tmp_path / "host", tmp_path / "engine"
        host.write_text(build + " host code")
        engine.write_text(build + " engine code")
        row["candidate"] = {"binary_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
                            "component_sha256": hashlib.sha256(engine.read_bytes()).hexdigest()}
        assert history(row, now=105)["history_identity"] == current
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pinned
    assert json.loads(path.read_text())["created_at"] == 100
    row["history_identity"] = identity("different tokens")
    with pytest.raises(ValueError, match="identity"):
        history(row, now=105)
    row["history_identity"] = current
    row["history_identity"]["harness_protocol"]["version"] = "different protocol"
    with pytest.raises(ValueError, match="identity"):
        history(row, now=105)
