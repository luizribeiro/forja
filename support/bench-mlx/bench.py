#!/usr/bin/env python3
"""Benchmark the bf16 MLX-LM baseline with Forja-compatible output."""

import argparse
import importlib.metadata
import json
import platform
import struct
import subprocess
import time
import tomllib
from pathlib import Path

MASK64 = (1 << 64) - 1


def arguments() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument(
        "--sampling", choices=("greedy", "generation-config"), default="greedy"
    )
    parser.add_argument("--json", required=True, type=Path)
    return parser.parse_args()


def load_suite(path: Path) -> dict[str, object]:
    """Read and validate the benchmark table shared with Forja."""
    with path.open("rb") as source:
        document = tomllib.load(source)
    if set(document) != {"bench"} or not isinstance(document["bench"], dict):
        raise ValueError("suite must contain only a [bench] table")
    bench = document["bench"]
    required = {
        "pp",
        "tg",
        "reps",
        "warmups",
        "decode_prefill",
        "contexts",
        "selection",
    }
    if set(bench) != required:
        raise ValueError(f"bench keys must be exactly {sorted(required)}")
    for key in ["pp", "tg", "reps", "decode_prefill"]:
        if isinstance(bench[key], bool) or not isinstance(bench[key], int) or bench[key] <= 0:
            raise ValueError(f"bench.{key} must be a positive integer")
    warmups = bench["warmups"]
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("bench.warmups must be a non-negative integer")
    contexts = bench["contexts"]
    if (
        not isinstance(contexts, list)
        or not contexts
        or any(isinstance(value, bool) or not isinstance(value, int) for value in contexts)
        or any(left >= right for left, right in zip(contexts, contexts[1:]))
    ):
        raise ValueError("bench.contexts must be non-empty and strictly increasing")
    selection = bench["selection"]
    allowed = {"host-argmax", "gpu-sequential", "gpu-pipelined"}
    if not isinstance(selection, list) or not selection or any(value not in allowed for value in selection):
        raise ValueError("bench.selection contains an unknown selection")
    return bench


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


def validate_bf16(model, mx, tree_flatten) -> None:
    """Require every floating model parameter to retain checkpoint bf16."""
    dtypes = {
        value.dtype
        for _, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating)
    }
    if dtypes != {mx.bfloat16}:
        raise RuntimeError(f"expected only bf16 model parameters, found {dtypes}")


def model_precision(model_dir: Path, model, mx, tree_flatten) -> str:
    """Validate and name the checkpoint precision used by MLX-LM."""
    config = json.loads((model_dir / "config.json").read_text())
    quantization = config.get("quantization_config")
    if quantization is not None:
        if quantization.get("bits") != 4:
            raise RuntimeError("expected an MLX affine 4-bit checkpoint")
        return "mlx-affine-q4"
    validate_bf16(model, mx, tree_flatten)
    return "bf16"


def sampling_options(model_dir: Path, mode: str) -> dict[str, object]:
    """Return the requested token-selection parameters."""
    if mode == "greedy":
        return {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0}
    config = json.loads((model_dir / "generation_config.json").read_text())
    if not config.get("do_sample"):
        raise RuntimeError("generation config does not enable sampling")
    return {
        "temperature": float32(config["temperature"]),
        "top_k": config["top_k"],
        "top_p": float32(config["top_p"]),
        "seed": 0,
    }


def float32(value: float) -> float:
    """Round a JSON number exactly as Forja's f32 configuration does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def model_overrides(model_dir: Path) -> dict[str, object]:
    """Fill required MLX-LM fields that Transformers supplies by default."""
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") == "olmoe" and config.get("rms_norm_eps") is None:
        return {"rms_norm_eps": 1e-5}
    return {}


def main() -> None:
    """Load the checkpoint, run MLX-LM generation trials, and write JSON."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    args = arguments()
    bench = load_suite(args.suite)
    model, tokenizer, config = load(
        args.model_dir,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        model_config=model_overrides(args.model_dir),
    )
    precision = model_precision(args.model_dir, model, mx, tree_flatten)
    mx.eval(model.parameters())
    tokenizer._eos_token_ids = set()
    sampling = sampling_options(args.model_dir, args.sampling)
    sampler = make_sampler(
        temp=sampling["temperature"],
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
    )
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    pp = bench["pp"]
    tg = bench["tg"]
    decode_prefill = bench["decode_prefill"]
    pp_prompt = synthetic_tokens(pp, vocab)
    tg_prompt = synthetic_tokens(decode_prefill, vocab)

    def pp_trial() -> float:
        mx.random.seed(sampling["seed"])
        response = next(
            stream_generate(
                model,
                tokenizer,
                pp_prompt,
                max_tokens=1,
                prefill_step_size=2048,
                sampler=sampler,
            )
        )
        return pp / response.prompt_tps

    def tg_trial() -> float:
        mx.random.seed(sampling["seed"])
        responses = iter(
            stream_generate(
                model,
                tokenizer,
                tg_prompt,
                max_tokens=tg + 2,
                prefill_step_size=2048,
                sampler=sampler,
            )
        )
        next(responses)
        next(responses)
        started = time.perf_counter()
        for _ in range(tg):
            next(responses)
        return time.perf_counter() - started

    def context_trial(context: int) -> float:
        mx.random.seed(sampling["seed"])
        prompt = synthetic_tokens(context - 1, vocab)
        responses = iter(
            stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=2,
                prefill_step_size=2048,
                sampler=sampler,
            )
        )
        next(responses)
        started = time.perf_counter()
        next(responses)
        return time.perf_counter() - started

    for _ in range(bench["warmups"]):
        pp_trial()
        tg_trial()
    pp_times = []
    tg_times = []
    for _ in range(bench["reps"]):
        pp_times.append(pp_trial())
        tg_times.append(tg_trial())
    context_times = []
    for context in bench["contexts"]:
        for _ in range(bench["warmups"]):
            context_trial(context)
        context_times.append(
            {
                "context_start": context,
                "token_generation": summary(
                    1, [context_trial(context) for _ in range(bench["reps"])]
                ),
            }
        )
    repository = Path(__file__).resolve().parents[2]
    device = mx.device_info()["device_name"]
    report = {
        "schema_version": 1,
        "implementation": "mlx_lm",
        "model": str(args.model_dir),
        "settings": {
            **bench,
            "selection": ["gpu-pipelined"],
            "sampling": sampling,
            "breakdown": True,
            "vary": {},
        },
        "tg_context_start": decode_prefill + 1,
        "results": [
            {
                "provenance": {
                    "git_commit": command_output(["git", "rev-parse", "HEAD"], repository),
                    "engine_component_sha256": None,
                    "device": device,
                    "os": f"macOS {platform.mac_ver()[0]}",
                    "precision": precision,
                    "mlx": mx.__version__,
                    "mlx_lm": importlib.metadata.version("mlx-lm"),
                },
                "prompt_processing": summary(pp, pp_times),
                "token_generation": summary(tg, tg_times),
                "context_token_generation": context_times,
            }
        ],
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    pp = report["results"][0]["prompt_processing"]["tokens_per_second"]["wall"]
    tg = report["results"][0]["token_generation"]["tokens_per_second"]["wall"]
    print(f"mlx {precision}\tpp\t{pp['median']:.2f} ({pp['ci95'][0]:.2f}–{pp['ci95'][1]:.2f})")
    print(f"mlx {precision}\ttg\t{tg['median']:.2f} ({tg['ci95'][0]:.2f}–{tg['ci95'][1]:.2f})")
    for result in context_times:
        rate = result["token_generation"]["tokens_per_second"]["wall"]
        print(
            f"mlx {precision}\ttg@{result['context_start']}\t"
            f"{rate['median']:.2f} ({rate['ci95'][0]:.2f}–{rate['ci95'][1]:.2f})"
        )


if __name__ == "__main__":
    main()
