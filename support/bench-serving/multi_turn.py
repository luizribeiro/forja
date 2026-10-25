#!/usr/bin/env python3
"""Measure a growing chat conversation against an OpenAI-compatible server."""

import argparse
import hashlib
import json
import shlex
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import http_client
import provenance
import serving_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_stats import stats  # noqa: E402

SUITE_KEYS = set(
    "schema_version model revision repetitions warmups connection_policy seed "
    "max_context output_tokens scripted_assistant_tokens tool_output_tokens".split()
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("forja", "mlx_lm"), required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--server-command", required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--forja-repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=serving_bench.artifact)
    parser.add_argument("--weights", action="append", default=[], type=Path)
    return parser.parse_args()


def load_suite(path: Path) -> tuple[dict[str, object], str]:
    data = path.read_bytes()
    document = tomllib.loads(data.decode())
    if set(document) != {"suite"} or set(document["suite"]) != SUITE_KEYS:
        raise ValueError("multi-turn suite has unknown or missing keys")
    suite = document["suite"]
    if suite["schema_version"] != 1 or suite["connection_policy"] != "reuse":
        raise ValueError("unsupported multi-turn suite")
    for key in ("repetitions", "max_context", "output_tokens", "scripted_assistant_tokens"):
        positive(suite[key], key)
    if not isinstance(suite["warmups"], int) or suite["warmups"] < 0:
        raise ValueError("warmups must be a nonnegative integer")
    tool_tokens = suite["tool_output_tokens"]
    if not isinstance(tool_tokens, list) or not tool_tokens:
        raise ValueError("tool_output_tokens must be a nonempty list")
    for value in tool_tokens:
        positive(value, "tool_output_tokens entry")
    return suite, hashlib.sha256(data).hexdigest()


def positive(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def user_content(repetition: str, turn: int) -> str:
    return f"Conversation {repetition}, turn {turn}. Continue the coding analysis after the tool result."


def turn_messages(repetition: str, turn: int, tool_tokens: int, assistant_tokens: int):
    call_id = f"call_{turn}"
    return [
        {"role": "user", "content": user_content(repetition, turn)},
        {
            "role": "assistant",
            "content": " result" * assistant_tokens,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": f'{{"turn":{turn}}}'}}],
        },
        {"role": "tool", "content": " tool" * tool_tokens, "tool_call_id": call_id},
    ]


def run_conversation(client, suite, repetition):
    messages = [{"role": "system", "content": f"Deterministic benchmark {repetition}."}]
    samples = []
    prompts = []
    for turn, tool_tokens in enumerate(suite["tool_output_tokens"], 1):
        messages.extend(turn_messages(repetition, turn, tool_tokens, suite["scripted_assistant_tokens"]))
        prompts.append({"messages": [dict(message) for message in messages], "max_tokens": suite["output_tokens"]})
        sample = client.chat(messages, suite["output_tokens"])
        if sample["error"] is not None:
            raise RuntimeError(f"turn {turn} failed: {sample['error']}")
        if not sample["itl_seconds"]:
            raise RuntimeError(f"turn {turn} produced fewer than two streamed tokens")
        total_context = sample["prompt_tokens"] + suite["output_tokens"]
        if total_context > suite["max_context"]:
            raise RuntimeError(f"turn {turn} needs {total_context} tokens, above max-context {suite['max_context']}")
        sample["total_context_tokens"] = total_context
        samples.append(sample)
    return samples, prompts


def turn_reports(runs: list[list[dict[str, object]]], tool_tokens: list[int]) -> list[dict[str, object]]:
    reports = []
    for index, added in enumerate(tool_tokens):
        samples = [run[index] for run in runs]
        decode_rates = [len(sample["itl_seconds"]) / sum(sample["itl_seconds"]) for sample in samples]
        reports.append(
            {
                "turn": index + 1,
                "cache_phase": "cold" if index == 0 else "cached",
                "tool_output_tokens_requested": added,
                "ttft_seconds": stats([sample["ttft_seconds"] for sample in samples]),
                "decode_tokens_per_second": stats(decode_rates),
                "prompt_tokens": stats([sample["prompt_tokens"] for sample in samples]),
                "output_tokens": stats([sample["output_tokens"] for sample in samples]),
                "cached_tokens": stats([sample["cached_tokens"] for sample in samples]),
                "total_context_tokens": stats([sample["total_context_tokens"] for sample in samples]),
                "samples": [
                    {
                        "ttft_seconds": sample["ttft_seconds"],
                        "decode_tokens_per_second": rate,
                        "prompt_tokens": sample["prompt_tokens"],
                        "output_tokens": sample["output_tokens"],
                        "cached_tokens": sample["cached_tokens"],
                        "total_context_tokens": sample["total_context_tokens"],
                        "output_sha256": hashlib.sha256(sample["output"].encode()).hexdigest(),
                    }
                    for sample, rate in zip(samples, decode_rates)
                ],
            }
        )
    return reports


def run(args: argparse.Namespace) -> dict[str, object]:
    suite, suite_hash = load_suite(args.suite)
    artifacts = dict(args.artifact)
    serving_bench.validate_identity(args, artifacts)
    command = shlex.split(args.server_command)
    server = serving_bench.Server(command, args.url)
    client = http_client.OpenAIClient(args.url, suite["model"])
    runs = []
    prompts = []
    peak_rss = 0
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        total = suite["warmups"] + suite["repetitions"]
        for repetition in range(total):
            measured = repetition >= suite["warmups"]
            label = f"sample-{repetition - suite['warmups']}" if measured else f"warmup-{repetition}"
            server.start()
            client.reset_connection()
            samples, bodies = run_conversation(client, suite, label)
            peak_rss = max(peak_rss, server.rss_bytes() or 0)
            if measured:
                runs.append(samples)
                prompts.extend(bodies)
    finally:
        server.stop()
    turns = turn_reports(runs, suite["tool_output_tokens"])
    for turn in turns:
        total = turn["total_context_tokens"]
        print(
            f"turn {turn['turn']} ({turn['cache_phase']}) total context: "
            f"median {total['median']:.0f}, max {total['ci95'][1]:.0f}"
        )
    return {
        "schema_version": 1,
        "implementation": args.implementation,
        "model": {"id": suite["model"], "revision": suite["revision"]},
        "suite_sha256": suite_hash,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "started_at": started_at,
        "duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "prompt_cache_reused": all(
            sample["cached_tokens"] > 0 for run in runs for sample in run[1:]
        ),
        "turns": turns,
        "provenance": provenance.collect(args.repository, args.forja_repository, artifacts, args.weights, args.resolved_config, command),
    }


def main() -> None:
    args = arguments()
    if args.json.exists():
        raise ValueError("refusing to overwrite an evidence artifact")
    record = run(args)
    args.json.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
