"""Run any JSON-producing competitor with cached measurements and drift probes."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

from bench_cache import Cache, digest


def file_hashes(paths: list[str]) -> str:
    """Hash actual model, workload or token files without trusting file dates."""
    if not paths:
        raise ValueError("identity file lists must not be empty")
    hashes = {}
    for name in sorted(paths):
        with Path(name).open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    return digest(hashes)


def measure(command: list[str]) -> dict:
    """Require a successful command writing a JSON object to {output}."""
    if (not command or any(not isinstance(arg, str) for arg in command)
            or command.count("{output}") != 1):
        raise ValueError("command must contain exactly one {output} argument")
    with tempfile.TemporaryDirectory(prefix="forja-competitor-") as folder:
        output = Path(folder) / "record.json"
        subprocess.run([str(output) if arg == "{output}" else arg for arg in command],
                       check=True)
        record = json.loads(output.read_text())
    if not isinstance(record, dict):
        raise ValueError("competitor must produce a JSON object")
    return record


def metric(record: dict, pointer: str) -> float:
    """Read a positive finite scalar using a JSON pointer."""
    if not pointer.startswith("/"):
        raise ValueError("metric must be a JSON pointer")
    value = record
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("drift metric must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("drift metric must be positive and finite")
    return value


def run(cache: Cache, full, probe, metrics: list[str], interval: float,
        tolerance: float, now: float | None = None, fresh: bool = False) -> dict:
    """Probe periodically; mismatch or drift forces a fresh full measurement."""
    if (not metrics or not math.isfinite(interval) or interval <= 0
            or not math.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("require metrics, positive drift interval and tolerance in (0,1)")
    now = time.time() if now is None else now
    entry = None if fresh else cache.read(now)
    if entry and now - entry["drift_checked_at"] >= interval:
        check = probe()
        ratios = {path: metric(check, path) / metric(entry["record"], path)
                  for path in metrics}
        passed = all(1 - tolerance <= ratio <= 1 + tolerance for ratio in ratios.values())
        drift = {"record": check, "ratios": ratios, "tolerance": tolerance,
                 "passed": passed, "checked_at": now}
        if passed:
            entry.update(drift_checked_at=now, drift=drift)
            cache.save(entry)
        else:
            cache.path.unlink(missing_ok=True)
            entry = cache.write(full(), now)
            entry["drift"] = drift
            cache.save(entry)
            return cache.report(entry, reused=False, now=now)
    if entry:
        return cache.report(entry, reused=True, now=now)
    return cache.report(cache.write(full(), now), reused=False, now=now)


def main() -> None:
    """Read a concrete request manifest; commands execute without a shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--max-age", type=float, default=7 * 86400)
    parser.add_argument("--drift-interval", type=float, default=86400)
    parser.add_argument("--drift-tolerance", type=float, default=.05)
    parser.add_argument("--fresh-competitor", action="store_true",
                        help="classic mode: ignore cached measurements and measure again")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    version = subprocess.run(request["version_command"], check=True, capture_output=True,
                             text=True).stdout.strip()
    identity = {"competitor_version": version,
                "effective_config": {"settings": request["effective_config"],
                                     "measure": request["measure"], "drift": request["drift"],
                                     "metrics": request["metrics"],
                                     "cache_runner_hash": file_hashes([
                                         str(Path(__file__)),
                                         str(Path(__file__).with_name("bench_cache.py"))])},
                **{f"{kind}_hash": file_hashes(request[f"{kind}_files"])
                   for kind in ["model", "workload", "tokens"]},
                "machine": {"hostname": platform.node(), "arch": platform.machine(),
                            "logical_cpus": os.cpu_count()},
                "os": {"platform": platform.platform(), "macos": platform.mac_ver()[0]}}
    cache = Cache(args.cache, identity, args.max_age)
    report = run(cache, lambda: measure(request["measure"]),
                 lambda: measure(request["drift"]), request["metrics"],
                 args.drift_interval, args.drift_tolerance, fresh=args.fresh_competitor)
    report["competitor_cache"]["mode"] = "fresh" if args.fresh_competitor else "cached-if-valid"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
