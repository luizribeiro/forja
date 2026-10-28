#!/usr/bin/env python3
"""Run the committed OpenAI serving suite against Forja or mlx-lm."""

import argparse
import hashlib
import http.client
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import http_client
import provenance
import priming
import suite


def arguments() -> argparse.Namespace:
    """Parse serving runner options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("forja", "mlx_lm"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--metrics-url")
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--forja-repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=artifact)
    parser.add_argument("--weights", action="append", default=[], type=Path)
    measurement_options(parser)
    return parser.parse_args()


def measurement_options(parser):
    """Expose fixed rep counts and raw windows to the interleaved gate driver."""
    parser.add_argument("--reps", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--raw-samples", action="store_true")
    parser.add_argument("--priming", choices=("legacy", "bounded"), default="legacy")


def measurement_settings(args, settings):
    settings = dict(settings)
    for flag, key, minimum in [("reps", "repetitions", 1), ("warmups", "warmups", 0)]:
        value = getattr(args, flag, None)
        if value is not None:
            if value < minimum:
                raise ValueError(f"invalid --{flag}")
            settings[key] = value
    return settings


def artifact(value: str) -> tuple[str, Path]:
    """Parse a named artifact path."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("artifact must be NAME=PATH")
    return name, Path(path)


def validate_identity(args: argparse.Namespace, artifacts: dict[str, Path]) -> None:
    """Refuse records missing implementation identity."""
    required = {"tokenizer"}
    if args.implementation == "forja":
        required |= {"binary", "component", "profile", "chat_format"}
        if args.resolved_config is None:
            raise ValueError("Forja records require --resolved-config")
    missing = required - artifacts.keys()
    if missing or not args.weights:
        raise ValueError(f"missing provenance artifacts: {sorted(missing or {'weights'})}")


class Server:
    """Owned server process restarted for cold cases."""

    def __init__(self, command: list[str], url: str):
        self.command = command
        self.url = url
        self.process: subprocess.Popen[bytes] | None = None
        self.startup_seconds = 0.0

    def start(self) -> None:
        """Start the server and wait for its model endpoint."""
        started = time.perf_counter()
        self.stop()
        self.process = subprocess.Popen(self.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}")
            if ready(self.url):
                self.startup_seconds += time.perf_counter() - started
                return
            time.sleep(0.1)
        raise TimeoutError("server did not become ready within 300 seconds")

    def stop(self) -> None:
        """Stop only the process started by this runner."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process = None

    def rss_bytes(self) -> int | None:
        """Sample resident bytes for the owned server."""
        if self.process is None:
            return None
        value = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.process.pid)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return int(value) * 1024 if value else None


def ready(base_url: str) -> bool:
    """Probe the common read-only model endpoint."""
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=1)
    try:
        connection.request("GET", parsed.path.rstrip("/") + "/v1/models")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def scrape_metrics(url: str | None) -> dict[str, float] | None:
    """Read unlabeled Prometheus samples from an optional endpoint."""
    if url is None:
        return None
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        connection.request("GET", parsed.path or "/metrics")
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 200:
            raise RuntimeError(f"metrics endpoint returned HTTP {response.status}")
    finally:
        connection.close()
    samples = {}
    for line in body.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        name, separator, value = line.partition(" ")
        if separator:
            samples[name] = float(value)
    return samples


def metric_report(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
) -> dict[str, object]:
    """Report resource gauges and counter deltas with explicit unavailable values."""
    memory = {
        name: None
        for name in ("pool_bytes", "table_bytes", "graph_bytes", "scratch_bytes")
    }
    phases = {
        name: None for name in ("scheduler_seconds", "queue_seconds", "parser_mask_seconds", "sse_seconds")
    }
    graphs = {"prepared": None, "preparation_seconds": None}
    report = {"gpu_seconds": None, "graphs": graphs, "memory": memory, "host_phases": phases}
    if before is None or after is None:
        return report
    for name in memory:
        metric = f"forja_{name}"
        if metric in after:
            memory[name] = int(after[metric])
    counters = {
        "scheduler_seconds": "forja_scheduler_nanoseconds_total",
        "queue_seconds": "forja_queue_nanoseconds_total",
        "parser_mask_seconds": "forja_parser_mask_nanoseconds_total",
        "sse_seconds": "forja_sse_nanoseconds_total",
    }
    for name, metric in counters.items():
        if metric in before and metric in after:
            phases[name] = max(0.0, after[metric] - before[metric]) / 1_000_000_000
    gpu = "forja_engine_gpu_time_nanoseconds"
    if gpu in before and gpu in after:
        report["gpu_seconds"] = max(0.0, after[gpu] - before[gpu]) / 1_000_000_000
    prepared = "forja_engine_prepared_graphs_total"
    if prepared in before and prepared in after:
        graphs["prepared"] = int(max(0.0, after[prepared] - before[prepared]))
    preparation = "forja_engine_graph_preparation_nanoseconds_total"
    if preparation in before and preparation in after:
        graphs["preparation_seconds"] = max(0.0, after[preparation] - before[preparation]) / 1_000_000_000
    return report


def exactness_report(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compare deterministic outputs at concurrency against their serial case."""
    serial = {}
    report = []
    for result in results:
        case = result["case"]
        if case["sampling"] not in {"greedy", "seeded"} or case.get("cancel_after_tokens"):
            continue
        identity = (
            case["name"],
            case["route"],
            case["prompt_tokens"],
            case["output_tokens"],
            case["sampling"],
            case.get("stream", False),
        )
        if case["concurrency"] == 1:
            serial[identity] = result["output_sha256"]
        elif identity in serial:
            report.append(
                {
                    "case": case["name"],
                    "prompt_tokens": case["prompt_tokens"],
                    "sampling": case["sampling"],
                    "concurrency": case["concurrency"],
                    "exact": result["output_sha256"] == serial[identity],
                }
            )
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run every expanded case and assemble one comparable record."""
    settings, cases, suite_hash = suite.load(args.suite)
    settings = measurement_settings(args, settings)
    concrete = suite.expand(cases)
    artifacts = dict(args.artifact)
    validate_identity(args, artifacts)
    command = shlex.split(args.server_command)
    server = Server(command, args.url)
    client: http_client.OpenAIClient | None = http_client.OpenAIClient(args.url, settings["model"])
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    results = []
    bounded = getattr(args, "priming", "legacy") == "bounded"
    primes = []
    cold_startup = 0.0
    timers = {"priming_seconds": 0.0, "prefix_preparation_seconds": 0.0, "measured_seconds": 0.0}
    try:
        for case in concrete:
            if (server.process is None and (bounded or not case.get("cold"))) or (bounded and case.get("cold")):
                server.start()
                client = http_client.OpenAIClient(args.url, settings["model"])
            assert client is not None
            if case.get("warm_prefix_tokens"):
                preparation_started = time.perf_counter()
                sample = client.request({**case, "prompt_tokens": 0, "output_tokens": 1})
                if bounded and sample["error"] is not None:
                    raise ValueError("warm-prefix preparation failed")
                timers["prefix_preparation_seconds"] += time.perf_counter() - preparation_started
            if bounded and settings["warmups"]:
                prime = priming.case(client, case)
                primes.append(prime)
                timers["priming_seconds"] += prime.get("seconds", 0.0)
            before = server.rss_bytes()
            restart = None
            metrics_before = None
            if case.get("cold"):
                def restart() -> None:
                    nonlocal cold_startup
                    previous = getattr(server, "startup_seconds", 0.0)
                    server.start()
                    cold_startup += getattr(server, "startup_seconds", 0.0) - previous
                    client.reset_connection()

            def begin_measurement() -> None:
                nonlocal metrics_before
                metrics_before = scrape_metrics(args.metrics_url)

            result = http_client.run_case(
                client,
                case,
                0 if bounded else settings["warmups"],
                settings["repetitions"],
                restart,
                begin_measurement,
                **({"raw_samples": True} if getattr(args, "raw_samples", False) else {}),
            )
            if bounded:
                priming.validate_case(case, result)
                timers["measured_seconds"] += result["measurement_seconds"]
            # A cold restart replaces the counters; cross-process differences are invalid.
            result["server_metrics"] = metric_report(
                None if bounded and case.get("cold") else metrics_before, scrape_metrics(args.metrics_url)
            )
            rss = [value for value in [before, server.rss_bytes()] if value is not None]
            result["rss_bytes"] = max(rss) if rss else None
            results.append(result)
    finally:
        server.stop()
    prompts = [http_client.request_body(case, settings["model"]) for case in concrete]
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if bounded:
        identity, timers["provenance_seconds"] = priming.timed_provenance(args, artifacts, command)
        measured_window = timers["measured_seconds"]
        timers["measured_seconds"] -= cold_startup
        timers["server_startup_seconds"] = server.startup_seconds
        timers = priming.phase_timers(started, timers, measured_seconds=measured_window,
                                     cold_startup_seconds=cold_startup)
    return {
        **({"priming": {"protocol": "bounded-path-v1", "paths": primes},
            "phase_timers": timers}
           if bounded else {}),
        "schema_version": 2,
        "measurement_mode": {"sampling": "fixed", "reps": settings["repetitions"],
                             "warmups": settings["warmups"], "unit": "whole-concurrent-case"},
        "implementation": args.implementation,
        "model": {"id": settings["model"], "revision": settings["revision"]},
        "suite_sha256": suite_hash,
        "prompt_sha256": prompt_hash,
        "sampling_seed": settings["seed"],
        "connection_policy": settings["connection_policy"],
        "warmups": settings["warmups"],
        "started_at": started_at,
        "duration_seconds": time.perf_counter() - started,
        "provenance": identity if bounded else provenance.collect(args.repository, args.forja_repository, artifacts, args.weights, args.resolved_config, command),
        "results": results,
        "exactness": exactness_report(results),
    }


def main() -> None:
    """Write one canonical JSON serving record."""
    args = arguments()
    record = run(args)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
