#!/usr/bin/env python3
"""Record a deterministic, random affine-q4 hybrid model and Transformers outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from fixture_library import GenerationSettings, manifest_base, prompt_tensors, sha256, versions
    from qwen38 import load_model
    torch.set_num_threads(2)
    torch.manual_seed(243)
    torch.use_deterministic_algorithms(True)
    config = Qwen3_5TextConfig(vocab_size=33, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=1, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
        layer_types=["linear_attention"]*3+["full_attention"],
        tie_word_embeddings=False, rope_parameters={"rope_type":"default", "rope_theta":10000000,
        "partial_rotary_factor":0.25, "mrope_section":[1,1,2], "mrope_interleaved": True})
    config._attn_implementation = "eager"
    with torch.device("meta"):
        prototype = Qwen3_5ForCausalLM(config)
    tensors = {}
    for name, parameter in prototype.named_parameters():
        name = "language_model." + name
        if parameter.ndim == 2:
            rows, columns = parameter.shape
            integers = torch.randint(0, 16, (rows, columns), dtype=torch.int64)
            words = (integers.reshape(rows, columns//8, 8) << (torch.arange(8)*4)).sum(-1).to(torch.uint32)
            tensors[name] = words
            tensors[name.removesuffix("weight")+"scales"] = torch.full((rows, columns//64), 0.0125, dtype=torch.bfloat16)
            tensors[name.removesuffix("weight")+"biases"] = torch.full((rows, columns//64), -0.09375, dtype=torch.bfloat16)
        else:
            value = torch.randn(parameter.shape) * 0.03
            if name.endswith("weight") and "conv1d" not in name:
                value += 1
            if name.endswith("dt_bias"):
                value -= 2
            if name.endswith("conv1d.weight"):
                value = (torch.randn(parameter.shape) * 0.2).transpose(-1,-2).contiguous()
            tensors[name] = value.to(torch.bfloat16)
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out / "model.safetensors"))
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{name:"model.safetensors" for name in tensors}},indent=2)+"\n")
    (args.out / "config.json").write_text(json.dumps({"text_config":config.to_dict()},indent=2)+"\n")
    model = load_model(args.out)
    blocks = {}
    for layer in [0,3]:
        name = "linear_attn" if layer == 0 else "self_attn"
        def hook(_module, inputs, kwargs, output, layer=layer):
            if f"block_input_{layer}" not in blocks:
                blocks[f"block_input_{layer}"] = (inputs[0] if inputs else kwargs["hidden_states"])[0].float().contiguous()
                blocks[f"block_output_{layer}"] = (output[0] if isinstance(output,tuple) else output)[0].float().contiguous()
        getattr(model.model.layers[layer],name).register_forward_hook(hook, with_kwargs=True)
    values = prompt_tensors(model, torch.tensor([[1,7,32]]), GenerationSettings())
    save_file(blocks, str(args.out / "blocks.safetensors"))
    values.update(blocks)
    save_file(values,str(args.out / "reference.safetensors"))
    manifest = manifest_base(args.out, sha256(args.out / "model.safetensors"), [("tiny", "fixed ids 1,7,32")], {})
    manifest["prompts"] = [{"name":"tiny", "text":"fixed ids 1,7,32", "file":"reference.safetensors", "sha256":sha256(args.out / "reference.safetensors"), "prompt_tokens":3}]
    (args.out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.out / "provenance.json").write_text(json.dumps({"libraries":versions(), "source_sha256":sha256(Path(__file__)), "reference_source_sha256":sha256(Path(__file__).with_name("qwen38.py")), "weights_sha256":sha256(args.out / "model.safetensors"), "reference_sha256":sha256(args.out / "reference.safetensors"), "seed":243,"device":"cpu"},indent=2)+"\n")


if __name__ == "__main__":
    main()
