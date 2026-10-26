#!/usr/bin/env python3
"""Measure streaming decodes while a cold long prompt joins the server."""

import hashlib
import json
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import http_client
import provenance
import serving_bench
import suite
from bench_stats import stats


def run_pair(client, decode, prefill):
    """Start the long prompt only after every decode has produced a token."""
    ready = threading.Barrier(decode["concurrency"] + 1, timeout=300)
    def request(case, callback=None):
        started = time.monotonic()
        sample = client.request(case, callback)
        return sample | {"started_monotonic": started, "ended_monotonic": time.monotonic()}

    with ThreadPoolExecutor(max_workers=decode["concurrency"] + 1) as executor:
        decodes = [executor.submit(request, decode, ready.wait)
                   for _ in range(decode["concurrency"])]
        ready.wait()
        background = executor.submit(request, prefill)
        samples = [future.result() for future in decodes]
        prompt = background.result()
    if any(sample["error"] for sample in [*samples, prompt]):
        raise RuntimeError("overlap request failed")
    if any(sample["ended_monotonic"] <= prompt["started_monotonic"] for sample in samples):
        raise RuntimeError("decode ended before the long prompt started")
    if prompt["cached_tokens"]:
        raise RuntimeError("long prompt unexpectedly reused a prefix")
    if any(len(sample["itl_seconds"]) < 2 for sample in samples):
        raise RuntimeError("decode ended before overlap could be measured")
    return samples, prompt


def run(args):
    """Retain the shared suite, requests, raw samples and engine provenance."""
    settings, cases, digest = suite.load(args.suite)
    prefill, *decodes = suite.expand(cases)
    if prefill["concurrency"] != 1 or any(not case.get("stream") for case in decodes):
        raise ValueError("overlap suite requires one prefill followed by streaming decodes")
    artifacts = dict(args.artifact)
    serving_bench.validate_identity(args, artifacts)
    command = shlex.split(args.server_command)
    server = serving_bench.Server(command, args.url)
    started = datetime.now(timezone.utc).isoformat()
    results = []
    try:
        for decode in decodes:
            runs = []
            for repetition in range(settings["warmups"] + settings["repetitions"]):
                server.start()
                client = http_client.OpenAIClient(args.url, settings["model"])
                samples, prompt = run_pair(client, decode, prefill)
                if repetition >= settings["warmups"]:
                    runs.append({"decodes": samples, "prefill": prompt})
                server.stop()
            intervals = [dt for run in runs for sample in run["decodes"] for dt in sample["itl_seconds"]]
            results.append({"case": decode, "itl_seconds": stats(intervals), "runs": runs})
    finally:
        server.stop()
    requests = [http_client.request_body(case, settings["model"]) for case in [prefill, *decodes]]
    return {
        "schema_version": 1, "implementation": args.implementation,
        "started_at": started, "model": {"id": settings["model"], "revision": settings["revision"]},
        "suite_sha256": digest, "warmups": settings["warmups"],
        "repetitions": settings["repetitions"],
        "prompt_sha256": hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
        "provenance": provenance.collect(args.repository, args.forja_repository, artifacts,
                                         args.weights, args.resolved_config, command),
        "results": results,
    }


if __name__ == "__main__":
    arguments = serving_bench.arguments()
    arguments.json.write_text(json.dumps(run(arguments), indent=2, sort_keys=True) + "\n")
