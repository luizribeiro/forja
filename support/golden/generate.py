#!/usr/bin/env python3
"""Generate deterministic Qwen3 reference fixtures with transformers."""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20250927
GENERATED_TOKENS = 32
PROMPTS = (
    ("short-english", "A quiet forge glows beneath the mountain."),
    (
        "code",
        "def checked_product(values):\n"
        "    result = 1\n"
        "    for value in values:\n"
        "        result *= value\n"
        "    return result\n",
    ),
    (
        "long-paragraph",
        "At dawn the research team crossed the empty courtyard and unlocked "
        "the laboratory. Their task was deliberately ordinary: measure each "
        "instrument, record every assumption, and repeat the experiment until "
        "small discrepancies had an explanation. One engineer calibrated the "
        "sensors while another compared yesterday's notes with the raw data. "
        "They resisted the temptation to smooth an awkward result, because an "
        "unexpected value can reveal more than a perfect chart. By noon they "
        "had traced a drifting measurement to a loose connector hidden behind "
        "the control panel. After replacing it, they ran the full sequence "
        "again with the same inputs and fixed timing. The second record agreed "
        "with the first everywhere except at the repaired channel. Before "
        "leaving, the team saved both records, wrote down the software versions, "
        "labeled the physical samples, and asked a colleague from another group "
        "to reproduce the calculation independently. The work felt slow, but "
        "the resulting evidence was clear enough that anyone could inspect it, "
        "challenge it, and obtain the same answer without relying on memory.",
    ),
    ("single-token", "Hello"),
)


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict[str, str]:
    """Return every runtime version that can affect fixture values."""
    return {
        "python": platform.python_version(),
        "safetensors": safetensors.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def manifest_base(model_path: Path, model_hash: str) -> dict[str, object]:
    """Build the invariant portion of the manifest."""
    return {
        "schema_version": 1,
        "model": {
            "directory": model_path.name,
            "file": "model.safetensors",
            "sha256": model_hash,
        },
        "libraries": versions(),
        "generation": {
            "attention": "eager",
            "device": "cpu",
            "dtype": "float32",
            "generated_tokens": GENERATED_TOKENS,
            "seed": SEED,
            "tokenization": "raw text, add_special_tokens=False",
        },
        "tensors": {
            "prompt_token_ids": "Tokenized raw prompt, shape [prompt_tokens], I64.",
            "hidden_state_0": "Token embeddings before decoder layer 1, shape [prompt_tokens, 1024], F32.",
            "hidden_state_i": "For 1 <= i < 28, output of decoder layer i before the final model norm, shape [prompt_tokens, 1024], F32.",
            "hidden_state_28": "Output of decoder layer 28 after the final model RMS norm, shape [prompt_tokens, 1024], F32.",
            "prompt_last_logits": "Logits at the last prompt position, shape [151936], F32.",
            "greedy_token_ids": "The 32 autoregressive argmax token ids, shape [32], I64.",
            "greedy_step_logits": "Logits used to select each greedy token, shape [32, 151936], F32.",
        },
        "prompts": [{"name": name, "text": text} for name, text in PROMPTS],
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


def set_determinism() -> None:
    """Configure deterministic CPU execution before loading the model."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def prompt_tensors(model, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run a prompt and its greedy continuation."""
    with torch.inference_mode():
        output = model(
            input_ids=token_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits[0, -1].float()
        tensors = {
            "prompt_token_ids": token_ids[0].contiguous(),
            "prompt_last_logits": logits.contiguous(),
        }
        for index, hidden_state in enumerate(output.hidden_states):
            tensors[f"hidden_state_{index}"] = hidden_state[0].float().contiguous()

        cache = output.past_key_values
        generated = []
        step_logits = []
        for step in range(GENERATED_TOKENS):
            next_token = logits.argmax(dim=-1)
            step_logits.append(logits)
            generated.append(next_token)
            if step + 1 < GENERATED_TOKENS:
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


def generate(model_path: Path, output: Path) -> None:
    """Generate every fixture and write the manifest last."""
    model_file = model_path / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_file}")
    model_hash = sha256(model_file)
    manifest = manifest_base(model_path, model_hash)
    if fixtures_are_current(output, manifest):
        print(f"fixtures are current: {output}")
        return

    set_determinism()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=None,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    fixture_prompts = []
    for name, text in PROMPTS:
        token_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        if name == "single-token" and token_ids.shape[1] != 1:
            raise ValueError(f"single-token prompt encoded as {token_ids.shape[1]} tokens")
        tensors = prompt_tensors(model, token_ids)
        if len([key for key in tensors if key.startswith("hidden_state_")]) != 29:
            raise ValueError("Qwen3-0.6B did not return 29 hidden states")
        fixture = output / f"{name}.safetensors"
        temporary = fixture.with_suffix(".safetensors.tmp")
        save_file(tensors, temporary, metadata={"prompt": name})
        temporary.replace(fixture)
        fixture_prompts.append(
            {
                "name": name,
                "text": text,
                "file": fixture.name,
                "sha256": sha256(fixture),
                "prompt_tokens": token_ids.shape[1],
            }
        )
        print(f"wrote {fixture} ({token_ids.shape[1]} prompt tokens)")

    manifest["prompts"] = fixture_prompts
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(output / "manifest.json")


def main() -> None:
    """Parse command-line arguments and generate fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    generate(args.model.resolve(), args.out.resolve())
    print(f"completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
