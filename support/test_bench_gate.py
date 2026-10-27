"""Exercise command persistence and provenance checks using synthetic native records."""

from copy import deepcopy
import json
import sys

import pytest

from gate_records import command_sampler
from gate_runner import run_row


def test_native_adapter_preserves_records_and_checks_hashes(tmp_path):
    script = tmp_path / "synthetic.py"
    script.write_text('''import json,sys,time
from pathlib import Path
now=time.time()
Path(sys.argv[1]).write_text(json.dumps({"implementation":"forja","provenance":{"binary_sha256":"host"},
"inputs":[{"engine_sha256":"guest"}],"config":"[bench]\\nreps="+sys.argv[2],
"results":[{"output_digest":"exact","score":100,"other_score":200,"window":[now,now+.0001]}]}))
''')
    adapter = {"command": [sys.executable, str(script), "{output}", "{reps}", "bench.warmups={warmups}"],
        "metric": "/results/0/score", "window": "/results/0/window",
        "expected_provenance": {"/provenance/binary_sha256": "host", "/inputs/0/engine_sha256": "guest"}}
    row = {"id": "row", "baseline": adapter, "candidate": deepcopy(adapter)}
    sample = command_sampler(row, tmp_path, [], 1, True)
    packet = sample("A", 1, "primary", 0, 0)
    assert packet["score"] == 100
    assert packet["output_digests"] == ["exact"]
    event = json.loads((tmp_path / "primary-00-0-A.command.json").read_text())
    assert event["command"][-1] == "bench.warmups=1"
    assert json.loads((tmp_path / "primary-00-0-A.json").read_text())["config"] == "[bench]\nreps=1"
    with pytest.raises(ValueError, match="repeat"):
        sample("A", 1, "primary", 0, 0)
    row["candidate"]["expected_provenance"]["/inputs/0/engine_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        sample("B", 1, "primary", 0, 1)


def test_shared_session_rows_reuse_observations_without_remeasuring(tmp_path):
    script = tmp_path / "session.py"
    script.write_text('''import json,math,sys,time
from pathlib import Path
now=time.time()
path=Path(sys.argv[1]);parts=path.stem.split("-")
shift=(.03 if int(parts[1])==0 else -.03 if int(parts[1])==1 else 0) if parts[-1]=="B" else 0
path.write_text(json.dumps({"score":100,"other":200*math.exp(shift),"window":[now,now+.0001]}))
''')
    adapter = {"command": [sys.executable, str(script), "{output}", "{reps}"],
               "metric": "/score", "window": "/window"}
    first = {"id": "first", "profile": "profile", "question": "equivalence",
             "baseline": adapter, "candidate": deepcopy(adapter)}
    memo = {}
    result = run_row(first, command_sampler(first, tmp_path, [], 1, True, memo), 123)
    assert result["decision"] == "no change"
    second = deepcopy(first)
    second["id"] = "second"
    second["baseline"]["metric"] = second["candidate"]["metric"] = "/other"
    result = run_row(second, command_sampler(second, tmp_path, [], 1, True, memo), 123)
    assert result["decision"] == "no change"
    assert len(result["runs"][0]["blocks"]) == 16
    assert all(s["reused_session"] for b in result["runs"][0]["blocks"][:6] for s in b["samples"])
    assert all(not s["reused_session"] for b in result["runs"][0]["blocks"][6:] for s in b["samples"])
    assert len(list(tmp_path.glob("*.command.json"))) == 64


def test_failed_worker_request_keeps_its_command_and_error(tmp_path):
    class FailedWorker:
        def sample(self, _side, _output, log, _phase, _block, _leg):
            log.write_text("worker failure")
            raise ValueError("lost response")
    adapter = {"command": ["/frozen/forja", "bench", "--reps", "{reps}", "--json", "{output}"]}
    row = {"id": "failed", "baseline": adapter, "candidate": adapter}
    sampler = command_sampler(row, tmp_path, [], 1, True, workers=FailedWorker())
    with pytest.raises(ValueError, match="lost response"):
        sampler("A", 1, "primary", 0, 0)
    event = json.loads((tmp_path / "primary-00-0-A.command.json").read_text())
    assert event["execution"] == "resident-worker" and event["returncode"] == -1
    assert event["error"] == {"type": "ValueError", "message": "lost response"}


def test_drift_promoted_rows_get_primary_measurements_after_their_screen(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    def fake(command, **_):
        now = time.time()
        from pathlib import Path
        Path(command[1]).write_text(json.dumps({"score": 100, "window": [now, now + .001]}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("gate_records.subprocess.run", fake)
    adapter = {"command": ["fake", "{output}", "{reps}"], "metric": "/score", "window": "/window"}
    row = {"id": "earlier", "baseline": adapter, "candidate": adapter}
    memo = {}
    earlier = command_sampler(row, tmp_path, [], 1, True, memo)("A", 1, "primary", 0, 0)
    promoted = command_sampler({**row, "id": "promoted"}, tmp_path, [], 1, True, memo)
    promoted("A", 1, "drift", 0, 0)
    fresh = promoted("A", 1, "primary", 0, 0)
    assert not fresh["reused_session"] and fresh["record_file"] != earlier["record_file"]
    assert "promoted-primary" in fresh["record_file"]
