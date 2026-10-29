# Inference engines

Start with the [engine guide](../docs/engines.md) and the
[SmolLM2 worked example](smollm2/README.md). Profiles pin model bytes, precision,
workload limits and device choices; each engine owns its tuning.

| Engine | Supported model | Profiles |
| --- | --- | --- |
| [SmolLM2](smollm2/) | SmolLM2-135M-Instruct | Reference f32, optimized f32 |
| [Qwen3](qwen3/) | Qwen3-0.6B | f32, bf16 |
| [OLMoE](olmoe/) | OLMoE-1B-7B-0924 | bf16 |
| [Qwen3-Coder](qwen3-coder/) | Qwen3-Coder-30B-A3B-Instruct | MLX 4-bit |

See [serving](../docs/serving.md) for configuration and
[benchmarking](../docs/benchmarking.md) for results and acceptance.
