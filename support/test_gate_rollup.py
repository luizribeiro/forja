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
