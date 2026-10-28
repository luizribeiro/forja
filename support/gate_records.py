"""Frozen per-leg records shared only within one gate invocation."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib
from bench_cache import digest
from bench_cache_run import metric, value_at

def save(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def verify_artifacts(artifacts: dict):
    for name, expected in artifacts.items():
        with Path(name).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"frozen artifact mismatch: {name}")

def measurement_command(adapter, output, reps, warmups, raw, guard):
    substitutions = {"{output}": str(output), "{reps}": str(reps), "{warmups}": str(warmups)}
    if "{output}" not in adapter["command"] or "{reps}" not in adapter["command"]:
        raise ValueError("measurement command must declare {output} and {reps}")
    return [*guard, *[substitutions.get(arg, arg).replace("{warmups}", str(warmups))
                     for arg in adapter["command"] if raw or arg != "--raw-samples"]]


def session_key(row):
    return digest([{key: row[side].get(key) for key in
        ["command", "cwd", "expected_provenance", "cache_state", "worker_protocol"]} for side in ["baseline", "candidate"]])


def command_sampler(row: dict, directory: Path, guard: list[str], warmups: int,
                    sequential: bool, memo: dict | None = None, workers=None):
    """Every original command, log and native record remains independently reviewable."""
    drift_seen = False
    def sample(side, reps, phase, block, leg):
        nonlocal drift_seen
        drift_seen = drift_seen or phase == "drift"
        adapter = row["baseline" if side == "A" else "candidate"]
        phase_name = phase
        if phase == "confirmation" or (phase == "primary" and drift_seen):
            label = "confirmation" if phase == "confirmation" else "promoted-primary"
            phase_name = f"{label}-{digest(row['id'])[:12]}"
        stem = f"{phase_name}-{block:02}-{leg}-{side}"
        output, log = directory / f"{stem}.json", directory / f"{stem}.log"
        raw = sequential or phase == "drift"
        command = measurement_command(adapter, output, reps, warmups, raw, guard)
        key = (phase_name, block, leg, side, reps)
        reused = memo is not None and key in memo
        if reused:
            event, expected_hash = memo[key]
            if event["command"] != command or event["cwd"] != adapter.get("cwd"):
                raise ValueError("shared session command differs")
            if hashlib.sha256(output.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("shared session record changed")
        else:
            if output.exists() or log.exists():
                raise ValueError("refusing to repeat an existing measurement")
            event = {"command": command, "cwd": adapter.get("cwd"), "started": time.time()}
            print(f"START {row['id']} {stem}", flush=True)
            if workers:
                try:
                    event.update(workers.sample(side, output, log, phase_name, block, leg))
                except Exception as error:
                    event.update(ended=time.time(), returncode=-1, execution="resident-worker",
                                 error={"type": type(error).__name__, "message": str(error)})
                    save(directory / f"{stem}.command.json", event)
                    raise
            else:
                with log.open("w") as stream:
                    result = subprocess.run(command, cwd=adapter.get("cwd"), stdout=stream, stderr=subprocess.STDOUT)
                event.update(ended=time.time(), returncode=result.returncode, execution="process-per-leg")
            save(directory / f"{stem}.command.json", event)
            if event["returncode"] != 0:
                raise subprocess.CalledProcessError(event["returncode"], command)
            if memo is not None:
                memo[key] = (event, hashlib.sha256(output.read_bytes()).hexdigest())
        record = json.loads(output.read_text())
        if workers and (record.get("worker", {}).get("pid") != event["worker_pid"]
                        or record["worker"].get("runner_loads") != event["worker_result"]["runner_loads"]
                        or record["worker"].get("instance_loads") != event["worker_result"].get("instance_loads")
                        or record.get("measurement_mode", {}).get("execution") != "resident-worker"):
            raise ValueError("native record does not match its resident worker")
        for pointer, expected in adapter.get("expected_provenance", {}).items():
            if value_at(record, pointer) != expected:
                raise ValueError(f"measurement provenance mismatch: {pointer}")
        if "reps" in record.get("measurement_mode", {}) and record["measurement_mode"]["reps"] != reps:
            raise ValueError("record repetition count differs from requested count")
        if record.get("implementation") == "forja" and "inputs" in record and "config" in record:
            required = {"/provenance/binary_sha256", "/inputs/0/engine_sha256"}
            if not required <= adapter.get("expected_provenance", {}).keys():
                raise ValueError("native adapter must pin binary and component hashes")
            if tomllib.loads(record["config"])["bench"]["reps"] != reps:
                raise ValueError("native repetition count differs from requested count")
            if any(not result.get("output_digest") for result in record["results"]):
                raise ValueError("native result has no output digest")
        if any(result.get("errors") for result in record.get("results", [])):
            raise ValueError("serving measurement contains errors")
        if record.get("implementation") == "forja" and any(not result["exact"] for result in record.get("exactness", [])):
            raise ValueError("serving exactness contract failed")
        if record.get("prompt_cache_reused") is False:
            raise ValueError("conversation cache path did not run")
        score = metric(record, adapter["metric"])
        if adapter.get("reciprocal", False):
            score = 1 / score
        window = value_at(record, adapter["window"]) if raw else [event["started"], event["ended"]]
        print(f"END {row['id']} {stem}", flush=True)
        return {"side": side, "start": window[0], "end": window[1], "score": score,
                "reused_session": reused, "execution": event.get("execution", "process-per-leg"),
                "worker_pid": event.get("worker_pid"),
                "output_digests": [result["output_digest"] for result in record.get("results", [])
                                   if result.get("output_digest")],
                "record_file": str(output), "record_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    return sample
