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
import suite


def arguments() -> argparse.Namespace:
    """Parse serving runner options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("forja", "mlx_lm"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--forja-repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=artifact)
    parser.add_argument("--weights", action="append", default=[], type=Path)
    return parser.parse_args()


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

    def start(self) -> None:
        """Start the server and wait for its model endpoint."""
        self.stop()
        self.process = subprocess.Popen(self.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}")
            if ready(self.url):
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


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run every expanded case and assemble one comparable record."""
    settings, cases, suite_hash = suite.load(args.suite)
    concrete = suite.expand(cases)
    artifacts = dict(args.artifact)
    validate_identity(args, artifacts)
    command = shlex.split(args.server_command)
    server = Server(command, args.url)
    client: http_client.OpenAIClient | None = http_client.OpenAIClient(args.url, settings["model"])
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    results = []
    try:
        for case in concrete:
            if server.process is None and not case.get("cold"):
                server.start()
                client = http_client.OpenAIClient(args.url, settings["model"])
            assert client is not None
            if case.get("warm_prefix_tokens"):
                client.request({**case, "prompt_tokens": 0, "output_tokens": 1})
            before = server.rss_bytes()
            restart = None
            if case.get("cold"):
                def restart() -> None:
                    server.start()
                    client.reset_connection()

            result = http_client.run_case(
                client,
                case,
                settings["warmups"],
                settings["repetitions"],
                restart,
            )
            rss = [value for value in [before, server.rss_bytes()] if value is not None]
            result["rss_bytes"] = max(rss) if rss else None
            results.append(result)
    finally:
        server.stop()
    prompts = [http_client.request_body(case, settings["model"]) for case in concrete]
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "implementation": args.implementation,
        "model": {"id": settings["model"], "revision": settings["revision"]},
        "suite_sha256": suite_hash,
        "prompt_sha256": prompt_hash,
        "sampling_seed": settings["seed"],
        "connection_policy": settings["connection_policy"],
        "warmups": settings["warmups"],
        "started_at": started_at,
        "duration_seconds": time.perf_counter() - started,
        "provenance": provenance.collect(args.repository, args.forja_repository, artifacts, args.weights, args.resolved_config, command),
        "results": results,
    }


def main() -> None:
    """Write one canonical JSON serving record."""
    args = arguments()
    record = run(args)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
