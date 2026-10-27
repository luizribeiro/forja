"""Native-protocol lifecycle tests use CPU-only fake workers."""
import json
import sys

import pytest

from gate_worker import Worker


def fake_worker(tmp_path, behavior="normal"):
    script = tmp_path / f"worker-{behavior}.py"
    script.write_text('''import json,os,sys
from pathlib import Path
print('startup log', flush=True)
print('FORJA_BENCH_READY '+json.dumps({'protocol':1,'pid':os.getpid(),'runner_loads':1,'instance_loads':{'single':1}}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 Path(request['output']).write_text(json.dumps({'pid':os.getpid(),'id':request['id']}))
 print('sample log',flush=True)
 print('FORJA_BENCH_RESULT '+json.dumps({'id':request['id'],'pid':os.getpid(),
  'runner_loads':LOADS,'instance_loads':{'single':1}}),flush=True)
'''.replace("LOADS", "2" if behavior == "reload" else "1"))
    return [sys.executable, str(script)]


def test_two_sides_keep_distinct_processes_across_many_requests(tmp_path):
    command = fake_worker(tmp_path)
    workers = [Worker(command, None, tmp_path / f"side-{side}.log", timeout=2) for side in "AB"]
    try:
        assert workers[0].ready["pid"] != workers[1].ready["pid"]
        for index, side in enumerate("ABBABAAB"):
            worker = workers[side == "B"]
            output = tmp_path / f"sample-{index}.json"
            event = worker.sample(output, tmp_path / f"sample-{index}.log")
            assert json.loads(output.read_text())["pid"] == worker.ready["pid"]
            assert event["worker_pid"] == worker.ready["pid"]
        assert all(worker.sequence == 4 for worker in workers)
    finally:
        for worker in workers:
            worker.close()
    assert all(worker.process.returncode == 0 for worker in workers)
    assert "sample log" in (tmp_path / "side-A.log").read_text()


def test_reloading_worker_fails_closed(tmp_path):
    worker = Worker(fake_worker(tmp_path, "reload"), None, tmp_path / "worker.log", timeout=2)
    try:
        with pytest.raises(ValueError, match="load count"):
            worker.sample(tmp_path / "record.json", tmp_path / "sample.log")
    finally:
        worker.close()


def test_worker_pair_preserves_workload_and_explicit_cold_process_fallback(tmp_path, monkeypatch):
    from gate_worker import worker_pair, close_workers
    class Fake:
        count = 0
        def __init__(self, command, cwd, log, timeout):
            Fake.count += 1
            self.ready = {"pid": Fake.count}
            self.startup = {"command": command}
            self.closed = False
        def close(self):
            self.closed = True
    monkeypatch.setattr("gate_worker.Worker", Fake)
    adapter = {"command": ["/frozen/forja", "bench", "--reps", "{reps}", "--set",
        "bench.warmups={warmups}", "--json", "{output}"], "expected_provenance": {
        "/provenance/binary_sha256": "binary", "/inputs/0/engine_sha256": "component"}}
    row = {"baseline": adapter, "candidate": dict(adapter)}
    workers, info = worker_pair(row, tmp_path, [], 1, 3)
    assert info["execution"] == "resident-worker"
    assert workers["A"].ready != workers["B"].ready
    command = workers["A"].startup["command"]
    assert command[command.index("--reps") + 1] == "1"
    assert "bench.warmups=3" in command and "--raw-samples" in command and "--worker" in command
    close_workers(workers)
    assert all(worker.closed for worker in workers.values())
    for enabled, policy, reason in [(False, "fresh-request", "explicit"), (True, "cold-process", "cold")]:
        row["candidate"]["cache_state"] = policy
        workers, info = worker_pair(row, tmp_path, [], 1, 3, enabled=enabled)
        assert not workers and reason in info["reason"]
    row["candidate"]["cache_state"] = "unknown"
    with pytest.raises(ValueError, match="cache-state"):
        worker_pair(row, tmp_path, [], 1, 3)
