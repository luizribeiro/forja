#!/usr/bin/env python3
"""CPU Transformers reference for the pinned affine-q4 text checkpoint."""
import argparse
import json
from pathlib import Path

from fixture_library import (GenerationSettings, manifest_base, prompt_tensors,
                             set_determinism, sha256)

PROMPTS = [("hello", "Hello"), ("arithmetic", "What is 2 + 2?"),
           ("completion", "The capital of France is")]


def load_model(root):
    """Adapt storage and affine projections; retain Transformers' model operations."""
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding)

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {file: safe_open(root / file, framework="pt", device="cpu")
              for file in set(index.values())}

    def tensor(name):
        return shards[index[name]].get_tensor(name)

    def dequantize(name, rows, columns):
        packed = tensor(name + ".weight")
        scales = tensor(name + ".scales").float()
        biases = tensor(name + ".biases").float()
        if list(packed.shape) != [rows, columns // 8] or list(scales.shape) != [rows, columns // 64] or biases.shape != scales.shape:
            raise ValueError(f"invalid affine projection: {name}")
        result = torch.empty(rows, columns, dtype=torch.float32)
        shifts = torch.arange(8, dtype=torch.int64) * 4
        for start in range(0, rows, 128):
            end = min(start + 128, rows)
            integers = ((packed[start:end].to(torch.int64).unsqueeze(-1) >> shifts) & 15).reshape(end-start, columns).float()
            result[start:end] = integers * scales[start:end].repeat_interleave(64, dim=1) + biases[start:end].repeat_interleave(64, dim=1)
        return result

    class AffineLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer("weight", weight)
        def forward(self, x):
            return torch.nn.functional.linear(x.float(), self.weight).to(x.dtype)

    class AffineEmbedding(AffineLinear):
        def forward(self, x):
            return torch.nn.functional.embedding(x, self.weight).to(torch.bfloat16)

    config = Qwen3_5TextConfig(**json.loads((root / "config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    config.tie_word_embeddings = False
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    norms = ("input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight", "q_norm.weight", "k_norm.weight")
    for name, parameter in list(model.named_parameters()):
        if name.endswith(".weight") and ("language_model." + name + "" ).removesuffix(".weight") + ".scales" in index:
            continue
        value = tensor("language_model." + name)
        if name.endswith("conv1d.weight"):
            value = value.transpose(-1, -2).contiguous()
        if name.endswith(norms):
            value = value.float() - 1.0
        if value.shape != parameter.shape:
            raise ValueError(f"parameter shape mismatch: {name}")
        model.get_submodule(name.rsplit(".", 1)[0]).register_parameter(name.rsplit(".", 1)[1], torch.nn.Parameter(value, requires_grad=False))
    for name, module in list(model.named_modules()):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            rows, columns = module.weight.shape
            cls = AffineEmbedding if isinstance(module, torch.nn.Embedding) else AffineLinear
            model.set_submodule(name, cls(dequantize("language_model." + name, rows, columns)))
            print(f"loaded {name}", flush=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cpu")
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise ValueError("uninitialized reference parameter")
    return model.eval()


def main():
    """Record independent hidden states and 32-token teacher-forced continuations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    torch.set_num_threads(16)
    settings = GenerationSettings(reference_weights="MLX affine q4 dequantized to f32; bf16 projection outputs")
    set_determinism(settings)
    root = args.model.resolve()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    model = load_model(root)
    manifest = manifest_base(root, sha256(root / "model.safetensors.index.json"), PROMPTS, {}, settings, model_file="model.safetensors.index.json")
    manifest["generation"]["dtype"] = "bf16"
    manifest["generation"]["quantization"] = {"bits": 4, "group_size": 64, "round_dequantized_weights": False}
    manifest["tokenizer_sha256"] = sha256(root / "tokenizer.json")
    manifest["source_sha256"] = sha256(Path(__file__))
    manifest["model"]["shards"] = {file: sha256(root / file) for file in sorted(set(json.loads((root / "model.safetensors.index.json").read_text())["weight_map"].values()))}
    args.out.mkdir(parents=True, exist_ok=True)
    prompts = []
    for name, text in PROMPTS:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        print(f"generating {name}: {ids}", flush=True)
        tensors = prompt_tensors(model, torch.tensor([ids]), settings)
        path = args.out / f"{name}.safetensors"
        save_file(tensors, str(path))
        prompts.append({"name": name, "text": text, "file": path.name, "sha256": sha256(path), "prompt_tokens": len(ids)})
        print(f"completed {name}: {tensors['greedy_token_ids'].tolist()}", flush=True)
    manifest["prompts"] = prompts
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
