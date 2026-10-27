"""Fail-closed verdict mixtures and a frozen-plan end-to-end execution."""

import hashlib
import json
import subprocess
import sys
import time

import pytest

import bench_gate
from gate_stats import METHOD


@pytest.mark.parametrize("decision,expected", [
    ("pass", "pass"), ("no change", "no change"), ("fixed-complete", "fixed-complete"),
    ("unchanged-drift-check", "pass"), ("inconclusive", "not-passed"),
    ("invalidated", "not-passed"), ("confirmed-fail", "not-passed"),
    ("unconfirmed-fail", "not-passed"), ("fail", "not-passed"),
    ("history-collected", "not-passed"),
])
def test_each_row_verdict_and_its_mixture_with_drift_checks(decision, expected):
    assert bench_gate.verdict([decision]) == expected
    assert bench_gate.verdict([decision, "unchanged-drift-check"]) == expected
    if decision != "pass" and decision != "unchanged-drift-check":
        assert bench_gate.verdict(["pass", decision]) == "not-passed"


def test_empty_and_mixed_diagnostic_or_fixed_verdicts_cannot_pass():
    with pytest.raises(ValueError, match="at least one"):
        bench_gate.verdict([])
    assert bench_gate.verdict(["no change", "fixed-complete"]) == "not-passed"


def test_main_checks_tree_artifacts_and_persists_verified_history(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.write_text("frozen")
    history = tmp_path / "noise.json"
    history.write_text(json.dumps({"identity": "workload", "method": METHOD,
        "created_at": time.time(), "log_scores": [-.003, .001, -.002, .002, .001, .003]}))
    adapter = {"command": ["fake", "{output}", "{reps}"]}
    plan = {"artifacts": {str(artifact): hashlib.sha256(artifact.read_bytes()).hexdigest()},
            "guard": [], "rows": [{"id": "row", "profile": "profile", "history_identity": "workload",
            "history": {"path": str(history), "sha256": hashlib.sha256(history.read_bytes()).hexdigest()},
            "baseline": adapter, "candidate": adapter}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    dirty = True
    calls = []
    monkeypatch.setattr(bench_gate.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(
        args, 0, stdout="dirty.txt" if dirty else ""))

    def factory(*_args):
        clock = 0.
        def sample(side, reps, phase, block, leg):
            nonlocal clock
            clock += 2
            calls.append((side, reps))
            return {"side": side, "start": clock, "end": clock + 1,
                    "score": 1 + (.001 * (block % 3) if side == "B" else 0)}
        return sample
    monkeypatch.setattr(bench_gate, "command_sampler", factory)
    monkeypatch.setattr(sys, "argv", ["gate", str(path), "--out", str(tmp_path / "dirty")])
    with pytest.raises(ValueError, match="clean"):
        bench_gate.main()
    dirty = False
    artifact.write_text("changed")
    monkeypatch.setattr(sys, "argv", ["gate", str(path), "--out", str(tmp_path / "mismatch")])
    with pytest.raises(ValueError, match="artifact"):
        bench_gate.main()
    assert not calls
    artifact.write_text("frozen")
    output = tmp_path / "valid"
    monkeypatch.setattr(sys, "argv", ["gate", str(path), "--out", str(output)])
    bench_gate.main()
    report = json.loads((output / "report.json").read_text())
    assert report["decision"] == "pass" and report["gating_pass"]
    saved = json.loads((output / "row-00/history.json").read_text())
    assert saved["identity"] == "workload" and saved["method"] == METHOD
    assert len(saved["log_scores"]) == 6 and saved["created_at"] > 0


def test_native_groups_reuse_blocks_with_fresh_processes_and_random_startup(tmp_path, monkeypatch):
    script = tmp_path / "native"
    script.write_text(f"#!{sys.executable}\n" + '''import json,os,sys,time
from pathlib import Path
print('FORJA_BENCH_READY '+json.dumps({'protocol':1,'pid':os.getpid(),'runner_loads':1,'instance_loads':{'single':1}}),flush=True)
for line in sys.stdin:
 request=json.loads(line);now=time.time()
 record={'implementation':'forja','provenance':{'binary_sha256':'host'},
  'inputs':[{'engine_sha256':'guest'}],'config':'[bench]\\nreps=1',
  'measurement_mode':{'execution':'resident-worker','reps':1},
  'worker':{'pid':os.getpid(),'runner_loads':1,'instance_loads':{'single':1}},
  'results':[{'output_digest':'exact','score':100,'window':[now,now+.000001]}]}
 Path(request['output']).write_text(json.dumps(record))
 print('FORJA_BENCH_RESULT '+json.dumps({'id':request['id'],'pid':os.getpid(),'runner_loads':1,'instance_loads':{'single':1}}),flush=True)
''')
    script.chmod(0o755)
    adapter = {"command": [str(script), "bench", "--reps", "{reps}", "--json", "{output}"],
        "metric": "/results/0/score", "window": "/results/0/window", "expected_provenance": {
        "/provenance/binary_sha256": "host", "/inputs/0/engine_sha256": "guest"}}
    other = {**adapter, "command": [*adapter["command"], "--set", "distinct-workload"]}
    plan = {"artifacts": {str(script): hashlib.sha256(script.read_bytes()).hexdigest()}, "guard": [],
        "rows": [{"id": name, "profile": "profile", "question": "equivalence",
                  "baseline": value, "candidate": value}
                 for name, value in [("first", adapter), ("other", other), ("shared", adapter)]]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(bench_gate.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 0, stdout=""))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["gate", str(path), "--out", str(output)])
    bench_gate.main()
    report = json.loads((output / "report.json").read_text())
    assert report["decision"] == "no change"
    assert [row["id"] for row in report["rows"]] == ["first", "shared", "other"]
    pids = [info["ready"]["pid"] for group in report["groups"]
            for block in group["workers"]["blocks"] for info in block["sides"].values()]
    assert len(set(pids)) == 24
    for group in report["groups"]:
        blocks = group["workers"]["blocks"]
        assert len(blocks) == 6
        for block in blocks:
            assert list(block["sides"]) == list(block["startup_order"])
            assert all(code == 0 for code in block["returncodes"].values())
            assert block["wall_seconds"] > 0 and block["completed"] and block["legs_completed"] == 4
        assert all(left["ended"] <= right["started"] for left, right in zip(blocks, blocks[1:]))
    assert report["mode"]["warmups_per_leg"] == 1 and report["mode"]["warmups_per_launch"] is None
    assert len(list(output.glob("group-*/*.command.json"))) == 48
    assert all(sample["reused_session"] for block in report["rows"][1]["runs"][0]["blocks"]
               for sample in block["samples"])
