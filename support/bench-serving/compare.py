#!/usr/bin/env python3
"""Compare only byte-for-byte compatible serving benchmark records."""

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    """Parse two record paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def comparable(left: dict[str, object], right: dict[str, object]) -> None:
    """Refuse every identity or workload mismatch before comparison."""
    checks = [
        ("schema version", left.get("schema_version"), right.get("schema_version")),
        ("suite hash", left.get("suite_sha256"), right.get("suite_sha256")),
        ("model id", left.get("model", {}).get("id"), right.get("model", {}).get("id")),
        ("model revision", left.get("model", {}).get("revision"), right.get("model", {}).get("revision")),
        ("prompt/request hash", left.get("prompt_sha256"), right.get("prompt_sha256")),
        ("sampling seed", left.get("sampling_seed"), right.get("sampling_seed")),
        ("warmups", left.get("warmups"), right.get("warmups")),
        ("connection policy", left.get("connection_policy"), right.get("connection_policy")),
    ]
    for name, old, new in checks:
        if old != new:
            raise ValueError(f"{name} differs")
    old_results = left.get("results", [])
    new_results = right.get("results", [])
    if len(old_results) != len(new_results):
        raise ValueError("request count differs")
    for old, new in zip(old_results, new_results):
        old_case = old.get("case", {})
        new_case = new.get("case", {})
        if old_case != new_case:
            for field in ("sampling", "concurrency"):
                if old_case.get(field) != new_case.get(field):
                    raise ValueError(f"{field} differs")
            raise ValueError("requests differ")


def rows(left: dict[str, object], right: dict[str, object]):
    """Yield comparable throughput and latency ratios."""
    for old, new in zip(left["results"], right["results"]):
        identity = old["case"]
        label = f"{identity['name']} pp{identity['prompt_tokens']} c{identity['concurrency']}"
        for metric in ("requests_per_second", "prompt_tokens_per_second", "output_tokens_per_second"):
            yield label, metric, ratio(new[metric], old[metric])
        for metric in ("ttft_seconds", "itl_seconds", "e2e_seconds"):
            if old[metric] is None or new[metric] is None:
                continue
            for percentile in ("p50", "p90", "p99"):
                yield label, f"{metric}.{percentile}", ratio(old[metric][percentile], new[metric][percentile])


def ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite comparison ratio when the baseline is nonzero."""
    return numerator / denominator if denominator else None


def main() -> None:
    """Validate records and print tab-separated ratios."""
    args = arguments()
    left = json.loads(args.baseline.read_text())
    right = json.loads(args.candidate.read_text())
    comparable(left, right)
    print("case\tmetric\tratio")
    for case, metric, value in rows(left, right):
        rendered = "n/a" if value is None else f"{value:.3f}x"
        print(case, metric, rendered, sep="\t")


if __name__ == "__main__":
    main()
