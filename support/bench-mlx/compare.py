#!/usr/bin/env python3
"""Print Forja and MLX-LM benchmark JSON side by side."""

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    """Parse report paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("forja", type=Path)
    parser.add_argument("mlx", type=Path)
    return parser.parse_args()


def rate(result: dict[str, object], metric: str) -> str:
    """Format a wall throughput median and confidence interval."""
    stats = result[metric]["tokens_per_second"]["wall"]
    return f"{stats['median']:.2f} ({stats['ci95'][0]:.2f}–{stats['ci95'][1]:.2f})"


def main() -> None:
    """Validate comparable settings and print one row per precision."""
    args = arguments()
    reports = [json.loads(args.forja.read_text()), json.loads(args.mlx.read_text())]
    if reports[0]["settings"] != reports[1]["settings"]:
        raise RuntimeError("benchmark settings differ")
    print("engine\tprecision\tpp wall tok/s (95% CI)\ttg wall tok/s (95% CI)")
    for report in reports:
        for result in report["results"]:
            print(
                report["implementation"],
                result["provenance"]["precision"],
                rate(result, "prompt_processing"),
                rate(result, "token_generation"),
                sep="\t",
            )


if __name__ == "__main__":
    main()
