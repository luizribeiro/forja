"""Serialized sample requests to one immutable, loaded native process."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time


class Worker:
    def __init__(self, command, cwd, log: Path, timeout=3600, startup_timeout=2700, protocol=1):
        self.command, self.cwd, self.timeout = command, cwd, timeout
        self.log = log.open("wb")
        self.buffer = b""
        self.sequence = 0
        self.ready = None
        started = time.time()
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.ready = self._response(b"FORJA_BENCH_READY ", startup_timeout)
            instances = self.ready.get("instance_loads", {})
            if (self.ready.get("protocol") != protocol or not instances
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
        if (result.get("protocol", 1) != self.ready["protocol"]
                or result.get("id") != self.sequence or result.get("pid") != self.ready["pid"]
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


def close_workers(workers):
    if hasattr(workers, "close"):
        workers.close()
    else:
        for worker in workers.values():
            worker.close()


def worker_policy(row, enabled=True):
    """Only direct pinned native commands promise the fixed-workload protocol."""
    adapters = [row[side] for side in ["baseline", "candidate"]]
    policies = {adapter.get("cache_state", "fresh-request") for adapter in adapters}
    if not policies <= {"fresh-request", "cold-process", "prefix-fork-per-worker"}:
        raise ValueError("unknown worker cache-state policy")
    protocols = {adapter.get("worker_protocol", 1) for adapter in adapters}
    if len(protocols) != 1 or not protocols <= {1, 2}:
        raise ValueError("worker protocols must match")
    if (protocols == {2}) != (policies == {"prefix-fork-per-worker"}):
        raise ValueError("worker protocol does not match cache-state policy")
    native = all(len(adapter["command"]) > 1 and adapter["command"][1] == "bench"
                 and {"/provenance/binary_sha256", "/inputs/0/engine_sha256"}
                     <= adapter.get("expected_provenance", {}).keys() for adapter in adapters)
    reason = ("explicit process-per-leg" if not enabled else "cold process required"
              if "cold-process" in policies else "adapter has no native worker protocol" if not native else None)
    return reason


def measurement_protocol(row, enabled, warmups):
    """Bind history to the effective execution and priming policy, including overrides."""
    adapters = [row[side] for side in ["baseline", "candidate"]]
    priming = {adapter["command"][adapter["command"].index("--priming") + 1]
               if "--priming" in adapter["command"] else "legacy" for adapter in adapters}
    if len(priming) != 1:
        raise ValueError("priming protocols must match")
    resident = worker_policy(row, enabled) is None
    protocol = adapters[0].get("worker_protocol", 1) if resident else None
    return {"execution": "resident-block-workers" if resident else "process-per-leg",
            "worker_protocol": protocol, "configured_priming_passes": warmups,
            "priming": "once-per-worker-shape; two decode steps" if protocol == 2 else priming.pop()}


def bind_measurement_protocol(row, enabled, warmups):
    """New protocols cannot borrow history from a different effective runtime mode."""
    modern = any(adapter.get("worker_protocol") == 2 or "--priming" in adapter["command"]
                 for adapter in [row["baseline"], row["candidate"]])
    if not modern or not row.get("history_identity"):
        return row
    actual = measurement_protocol(row, enabled, warmups)
    identity = row["history_identity"]
    if row.get("history") and identity.get("measurement_protocol") != actual:
        raise ValueError("history measurement protocol differs from runtime mode")
    return {**row, "history_identity": {**identity, "measurement_protocol": actual}}


def worker_pair(row, directory, guard, reps, warmups, enabled=True, timeout=3600, order="AB"):
    from gate_records import measurement_command, save
    reason = worker_policy(row, enabled)
    if reason:
        return {}, {"execution": "process-per-leg", "reason": reason}
    if len(order) != 2 or set(order) != set("AB"):
        raise ValueError("worker startup order must permute sides A and B")
    workers = {}
    try:
        for side in order:
            adapter = row["baseline" if side == "A" else "candidate"]
            command = measurement_command(adapter, directory / f"worker-{side}-unused.json",
                                          reps, warmups, True, guard)
            workers[side] = Worker(worker_command(command), adapter.get("cwd"),
                                   directory / f"worker-{side}.log", timeout=timeout,
                                   protocol=adapter.get("worker_protocol", 1))
            save(directory / f"worker-{side}.start.json", workers[side].startup)
        if workers["A"].ready["pid"] == workers["B"].ready["pid"]:
            raise ValueError("baseline and candidate workers must be separate processes")
    except BaseException:
        close_workers(workers)
        raise
    return workers, {"execution": "resident-worker", "cache_state": row["baseline"].get("cache_state", "fresh-request"), "startup_order": order,
                     "sides": {side: worker.startup for side, worker in workers.items()}}
