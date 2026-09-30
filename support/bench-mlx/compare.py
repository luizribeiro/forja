#!/usr/bin/env python3
"""Print Forja and MLX-LM benchmark JSON side by side."""

import argparse
import json
import tomllib
from pathlib import Path


def arguments() -> argparse.Namespace:
    """Parse report paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("forja", type=Path)
    parser.add_argument("mlx", type=Path)
    return parser.parse_args()


def rate(report: dict[str, object], result: dict[str, object], metric: str) -> str:
    """Format a wall throughput median and confidence interval."""
    field = metric if report.get("schema_version") == 2 else {
        "pp": "prompt_processing",
        "tg": "token_generation",
    }[metric]
    stats = result[field]["tokens_per_second"]["wall"]
    return f"{stats['median']:.2f} ({stats['ci95'][0]:.2f}–{stats['ci95'][1]:.2f})"


def settings(report: dict[str, object]) -> dict[str, object]:
    """Return benchmark settings from either record schema."""
    if report.get("schema_version") == 2:
        values = tomllib.loads(report["config"])["bench"]
    else:
        values = report["settings"].copy()
        for old, new in {
            "prompt_tokens": "pp",
            "generated_tokens": "tg",
            "repetitions": "reps",
        }.items():
            if old in values:
                values[new] = values.pop(old)
    if values.get("breakdown") is False:
        values.pop("breakdown")
    if values.get("vary") == {}:
        values.pop("vary")
    return values


def engine(report: dict[str, object], result: dict[str, object]) -> str:
    """Return a path-free engine label from either record schema."""
    if report.get("schema_version") == 2:
        digest = report["inputs"][result["input"]]["engine_sha256"]
        return digest[:12]
    provenance = result["provenance"]
    return provenance.get("engine", provenance.get("precision", "unknown"))


def main() -> None:
    """Validate comparable settings and print one row per engine."""
    args = arguments()
    reports = [json.loads(args.forja.read_text()), json.loads(args.mlx.read_text())]
    if settings(reports[0]) != settings(reports[1]):
        raise RuntimeError("benchmark settings differ")
    depths = [
        report.get("tg_context_start", settings(report).get("decode_prefill", 0) + 1)
        for report in reports
    ]
    if depths[0] != depths[1]:
        raise RuntimeError("token-generation context depths differ")
    print("implementation\tengine\tpp wall tok/s (95% CI)\ttg wall tok/s (95% CI)")
    for report in reports:
        for result in report["results"]:
            print(
                report["implementation"],
                engine(report, result),
                rate(report, result, "pp"),
                rate(report, result, "tg"),
                sep="\t",
            )
    print("\nimplementation\tengine\tcontext\ttg wall tok/s (95% CI)")
    for report in reports:
        for result in report["results"]:
            if report.get("schema_version") == 2:
                contexts = [
                    {
                        "context_start": value["context_start"],
                        "token_generation": value["unprofiled_token_generation"],
                    }
                    for value in result.get("breakdown", [])
                ]
            else:
                contexts = result.get("context_token_generation", [])
            for context in contexts:
                print(
                    report["implementation"],
                    engine(report, result),
                    context["context_start"],
                    rate(
                        {"schema_version": 1},
                        context,
                        "tg",
                    ),
                    sep="\t",
                )


if __name__ == "__main__":
    main()
