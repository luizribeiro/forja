"""Build deterministic transformer reference fixtures.

To debug a layer reported by ``forja verify``, attach ``register_forward_hook``
to ``model.model.layers[i].self_attn``, ``.mlp``, or the MoE ``.mlp.gate`` and
save the hook output for the same prompt as a contiguous float32 safetensor.
Compare that dump with the engine tap using normwise relative error, moving the
hook inward until the first mismatch is isolated. Transformers returns the last
hidden state after the final norm; router logits are captured before top-k.
"""

import hashlib
import json
import os
import platform
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    """Settings that affect deterministic greedy generation."""

    seed: int = 20250927
    generated_tokens: int = 32
    num_hidden_layers: int | None = None

    def __post_init__(self) -> None:
        """Reject layer limits that cannot describe a model."""
        if self.num_hidden_layers is not None and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")


Prompt = tuple[str, str]
FixtureValidator = Callable[[str, Any, dict[str, Any]], None]
UNSAFE_WEIGHT_SUFFIXES = frozenset({".bin", ".pt", ".pth"})


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    """Return every runtime version that can affect fixture values."""
    import safetensors
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "safetensors": safetensors.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def manifest_base(
    model_path: Path,
    model_hash: str,
    prompts: Sequence[Prompt],
    tensor_descriptions: dict[str, str],
    settings: GenerationSettings | None = None,
    *,
    model_file: str = "model.safetensors",
) -> dict[str, object]:
    """Build the invariant portion of a fixture manifest."""
    settings = settings or GenerationSettings()
    generation = {
        "attention": "eager",
        "device": "cpu",
        "dtype": "float32",
        "generated_tokens": settings.generated_tokens,
        "seed": settings.seed,
        "tokenization": "raw text, add_special_tokens=False",
    }
    if settings.num_hidden_layers is not None:
        generation["num_hidden_layers"] = settings.num_hidden_layers
    return {
        "schema_version": 1,
        "model": {
            "directory": model_path.name,
            "file": model_file,
            "sha256": model_hash,
        },
        "libraries": versions(),
        "generation": generation,
        "tensors": tensor_descriptions,
        "prompts": [{"name": name, "text": text} for name, text in prompts],
    }


def fixtures_are_current(output: Path, expected: dict[str, object]) -> bool:
    """Check the manifest identity and all fixture hashes."""
    path = output / "manifest.json"
    try:
        actual = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    expected_without_files = dict(expected)
    actual_without_files = dict(actual)
    prompts = actual_without_files.pop("prompts", [])
    actual_without_files["prompts"] = [
        {"name": prompt.get("name"), "text": prompt.get("text")} for prompt in prompts
    ]
    if actual_without_files != expected_without_files:
        return False
    for prompt in prompts:
        fixture = output / prompt.get("file", "")
        if not fixture.is_file() or sha256(fixture) != prompt.get("sha256"):
            return False
    return True


def validate_model_directory(model_path: Path) -> None:
    """Refuse model files that can execute code or use unsafe serialization."""
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in UNSAFE_WEIGHT_SUFFIXES:
            raise ValueError(f"unsafe model weight file: {path}")
        if path.suffix.lower() == ".py":
            raise ValueError(f"model directory contains Python code: {path}")

    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("auto_map") is not None:
        raise ValueError(f"model config requires remote code: {config_path}")

    import transformers

    architectures = config.get("architectures", [])
    unknown = [
        architecture
        for architecture in architectures
        if not isinstance(architecture, str)
        or getattr(transformers, architecture, None) is None
    ]
    if unknown:
        raise ValueError(f"model config names custom architectures: {unknown}")


def model_weight_file(model_path: Path) -> Path:
    """Return the single weight file or validate and return a shard index."""
    single = model_path / "model.safetensors"
    if single.is_file():
        return single

    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors shard index: {index_path}")

    from safetensors import safe_open

    shards: dict[str, set[str]] = {}
    for tensor, file_name in weight_map.items():
        if (
            not isinstance(tensor, str)
            or not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not file_name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid safetensors shard index: {index_path}")
        shards.setdefault(file_name, set()).add(tensor)
    for file_name, expected_tensors in sorted(shards.items()):
        shard = model_path / file_name
        if not shard.is_file():
            raise FileNotFoundError(f"model shard does not exist: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as source:
            actual_tensors = set(source.keys())
        if actual_tensors != expected_tensors:
            raise ValueError(f"model shard does not match index: {shard}")
    return index_path


def set_determinism(settings: GenerationSettings) -> None:
    """Configure deterministic CPU execution before loading the model."""
    import torch

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(True)


def prompt_tensors(
    model: Any,
    token_ids: Any,
    settings: GenerationSettings,
    *,
    capture_router_logits: bool = False,
) -> dict[str, Any]:
    """Run a prompt and its greedy continuation."""
    import torch

    with torch.inference_mode():
        prompt_options = {
            "input_ids": token_ids,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
        if capture_router_logits:
            prompt_options["output_router_logits"] = True
        output = model(
            **prompt_options,
        )
        logits = output.logits[0, -1].float()
        tensors = {
            "prompt_token_ids": token_ids[0].contiguous(),
            "prompt_last_logits": logits.contiguous(),
        }
        for index, hidden_state in enumerate(output.hidden_states):
            tensors[f"hidden_state_{index}"] = hidden_state[0].float().contiguous()
        if capture_router_logits:
            tensors["router_logits"] = torch.stack(
                [
                    logits.reshape(token_ids.shape[1], -1)
                    for logits in output.router_logits
                ]
            ).float().contiguous()

        cache = output.past_key_values
        generated = []
        step_logits = []
        for step in range(settings.generated_tokens):
            next_token = logits.argmax(dim=-1)
            step_logits.append(logits)
            generated.append(next_token)
            if step + 1 < settings.generated_tokens:
                output = model(
                    input_ids=next_token.reshape(1, 1),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                logits = output.logits[0, -1].float()
                cache = output.past_key_values
        tensors["greedy_token_ids"] = torch.stack(generated).to(torch.int64)
        tensors["greedy_step_logits"] = torch.stack(step_logits)
    return tensors


def generate(
    model_path: Path,
    output: Path,
    prompts: Sequence[Prompt],
    tensor_descriptions: dict[str, str],
    *,
    settings: GenerationSettings | None = None,
    validate_fixture: FixtureValidator | None = None,
) -> None:
    """Generate fixtures for every prompt and write the manifest last."""
    settings = settings or GenerationSettings()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    validate_model_directory(model_path)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_file = model_weight_file(model_path)
    model_hash = sha256(model_file)
    manifest = manifest_base(
        model_path,
        model_hash,
        prompts,
        tensor_descriptions,
        settings,
        model_file=model_file.name,
    )
    if fixtures_are_current(output, manifest):
        print(f"fixtures are current: {output}")
        return

    set_determinism(settings)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if settings.num_hidden_layers is not None:
        config.num_hidden_layers = settings.num_hidden_layers
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float32,
        device_map=None,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    capture_router_logits = getattr(model.config, "num_experts", 0) > 0
    output.mkdir(parents=True, exist_ok=True)
    fixture_prompts = []
    for name, text in prompts:
        token_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        tensors = prompt_tensors(
            model,
            token_ids,
            settings,
            capture_router_logits=capture_router_logits,
        )
        if validate_fixture is not None:
            validate_fixture(name, token_ids, tensors)
        fixture = output / f"{name}.safetensors"
        temporary = fixture.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary, metadata={"prompt": name})
        temporary.replace(fixture)
        prompt_manifest = {
            "name": name,
            "text": text,
            "file": fixture.name,
            "sha256": sha256(fixture),
            "prompt_tokens": token_ids.shape[1],
        }
        if capture_router_logits:
            prompt_manifest["router_logits"] = {
                "shape": list(tensors["router_logits"].shape)
            }
        fixture_prompts.append(prompt_manifest)
        print(f"wrote {fixture} ({token_ids.shape[1]} prompt tokens)")

    manifest["prompts"] = fixture_prompts
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(output / "manifest.json")
