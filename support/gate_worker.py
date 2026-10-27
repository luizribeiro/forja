"""Serialized sample requests to one immutable, loaded native process."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


class Worker:
    def __init__(self, command, cwd, log: Path, timeout=3600, startup_timeout=2700):
        self.command, self.cwd, self.timeout = command, cwd, timeout
        self.log = log.open("wb")
        self.buffer = b""
        self.sequence = 0
        self.ready = None
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        started = time.time()
        try:
            self.ready = self._response(b"FORJA_BENCH_READY ", startup_timeout)
            instances = self.ready.get("instance_loads", {})
            if (self.ready.get("protocol") != 1 or not instances
                    or any(count != 1 for count in instances.values())
                    or self.ready.get("runner_loads") != sum(instances.values())):
                raise ValueError("worker must load each isolated measurement instance exactly once")
        except BaseException:
            self.close()
            raise
        self.startup = {"started": started, "ended": time.time(), "command": command,
                        "cwd": cwd, "ready": self.ready}

    def _response(self, prefix, timeout, tee=None):
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.startswith(prefix):
                    return json.loads(line[len(prefix):])
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("resident worker response timed out")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise ValueError("resident worker exited before its response")
            self.log.write(data)
            self.log.flush()
            if tee is not None:
                tee.write(data)
            self.buffer += data

    def sample(self, output: Path, log: Path):
        if output.exists():
            raise ValueError("worker output already exists")
        self.sequence += 1
        request = {"id": self.sequence, "output": str(output.resolve())}
        started = time.time()
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        with log.open("wb") as stream:
            result = self._response(b"FORJA_BENCH_RESULT ", self.timeout, stream)
        if (result.get("id") != self.sequence or result.get("pid") != self.ready["pid"]
                or result.get("runner_loads") != self.ready["runner_loads"]
                or result.get("instance_loads") != self.ready["instance_loads"]):
            raise ValueError("worker response identity or load count changed")
        if not output.is_file():
            raise ValueError("worker completed without a native record")
        return {"started": started, "ended": time.time(), "returncode": 0,
                "execution": "resident-worker", "worker_pid": self.ready["pid"],
                "worker_request": request, "worker_result": result,
                "worker_command": self.command}

    def close(self):
        if self.log.closed:
            return
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            if self.ready is not None:
                os.killpg(self.ready["pid"], signal.SIGTERM)
            else:
                subprocess.run(["pkill", "-TERM", "-P", str(self.process.pid)], check=False)
            self.process.terminate()
            self.process.wait(timeout=60)
        remaining = self.process.stdout.read()
        if remaining:
            self.log.write(remaining)
        self.process.stdout.close()
        self.selector.close()
        self.log.close()


def worker_command(command):
    """Keep the frozen workload, including repetitions and warmups, at launch."""
    return [*command, "--worker", *([] if "--raw-samples" in command else ["--raw-samples"])]
