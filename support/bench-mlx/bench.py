#!/usr/bin/env python3
"""Benchmark the bf16 MLX-LM baseline with Forja-compatible output."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate

WARMUPS = 3
MASK64 = (1 << 64) - 1


def arguments() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--pp", default=512, type=positive)
    parser.add_argument("--tg", default=128, type=positive)
    parser.add_argument("--reps", default=30, type=positive)
    parser.add_argument("--json", required=True, type=Path)
    return parser.parse_args()


def positive(value: str) -> int:
    """Parse a positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def synthetic_tokens(count: int, vocab: int) -> list[int]:
    """Return the same fixed-seed token ids used by Forja."""
    state = 0x4D595DF4D0F33173
    tokens = []
    for _ in range(count):
        state = (state * 6_364_136_223_846_793_005 + 1) & MASK64
        tokens.append(state % vocab)
    return tokens


def stats(values: list[float]) -> dict[str, object]:
    """Return a median and distribution-free 95% order-statistic interval."""
    values = sorted(values)
    count = len(values)
    median = (
        values[count // 2]
        if count % 2
        else (values[count // 2 - 1] + values[count // 2]) / 2
    )
    probability = 0.5**count
    cumulative = probability
    rank = 0
    for index in range(count // 2):
        probability *= (count - index) / (index + 1)
        if cumulative + probability > 0.025:
            break
        cumulative += probability
        rank = index + 1
    return {"median": median, "ci95": [values[rank], values[-rank - 1]]}


def summary(tokens: int, times: list[float]) -> dict[str, object]:
    """Build one timing summary in the shared schema."""
    return {
        "tokens_per_second": {
            "wall": stats([tokens / elapsed for elapsed in times]),
            "gpu": None,
        },
        "wall_time_seconds": stats(times),
        "gpu_time_seconds": None,
        "submissions": None,
    }


def command_output(arguments: list[str], cwd: Path | None = None) -> str:
    """Run a provenance command and require nonempty output."""
    value = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not value:
        raise RuntimeError(f"{arguments[0]} returned no output")
    return value


def validate_bf16(model) -> None:
    """Require every floating model parameter to retain checkpoint bf16."""
    dtypes = {
        value.dtype
        for _, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating)
    }
    if dtypes != {mx.bfloat16}:
        raise RuntimeError(f"expected only bf16 model parameters, found {dtypes}")


def main() -> None:
    """Load the checkpoint, run MLX-LM generation trials, and write JSON."""
    args = arguments()
    model, tokenizer, config = load(
        args.model_dir,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
    )
    validate_bf16(model)
    mx.eval(model.parameters())
    tokenizer._eos_token_ids = set()
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = synthetic_tokens(args.pp, vocab)

    def trial():
        response = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.tg,
            prefill_step_size=2048,
        ):
            pass
        if response is None or response.generation_tokens != args.tg:
            raise RuntimeError("MLX-LM generation ended before the requested length")
        return response

    for _ in range(WARMUPS):
        trial()
    responses = [trial() for _ in range(args.reps)]
    pp_times = [args.pp / response.prompt_tps for response in responses]
    tg_times = [args.tg / response.generation_tps for response in responses]
    repository = Path(__file__).resolve().parents[2]
    device = mx.device_info()["device_name"]
    report = {
        "schema_version": 1,
        "implementation": "mlx_lm",
        "model": str(args.model_dir),
        "settings": {
            "prompt_tokens": args.pp,
            "generated_tokens": args.tg,
            "warmups": WARMUPS,
            "repetitions": args.reps,
        },
        "results": [
            {
                "provenance": {
                    "git_commit": command_output(["git", "rev-parse", "HEAD"], repository),
                    "engine_component_sha256": None,
                    "device": device,
                    "os": f"macOS {platform.mac_ver()[0]}",
                    "precision": "bf16",
                    "mlx": mx.__version__,
                    "mlx_lm": importlib.metadata.version("mlx-lm"),
                },
                "prompt_processing": summary(args.pp, pp_times),
                "token_generation": summary(args.tg, tg_times),
            }
        ],
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    pp = report["results"][0]["prompt_processing"]["tokens_per_second"]["wall"]
    tg = report["results"][0]["token_generation"]["tokens_per_second"]["wall"]
    print(f"mlx bf16\tpp\t{pp['median']:.2f} ({pp['ci95'][0]:.2f}–{pp['ci95'][1]:.2f})")
    print(f"mlx bf16\ttg\t{tg['median']:.2f} ({tg['ci95'][0]:.2f}–{tg['ci95'][1]:.2f})")


if __name__ == "__main__":
    main()
