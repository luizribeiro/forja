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
