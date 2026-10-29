# Forja

**Fast, capability-safe LLM inference on Apple Silicon.**

Forja is a Rust host that runs inference engines as WebAssembly components and
executes their GPU work through Metal 4. Engines own their architectures and
tuning; small, import-free chat components own message formatting.

- **Safety by construction.** A restricted kernel language and validated buffer
  capabilities bound GPU memory access. Guests never submit Metal source.
- **Measured speed.** [`forja bench` records](engines/smollm2/bench/) preserve model,
  component, configuration and machine provenance; the [worked example](engines/smollm2/README.md)
  includes measured optimization gains and an `mlx_lm` comparison.
- **Engine-owned tuning.** Profiles select kernels, fusion and replay without
  model or chip switches in the host.

| Supported model | Profiles and usage |
| --- | --- |
| SmolLM2-135M-Instruct | [Reference and optimized f32](engines/smollm2/) |
| Qwen3-0.6B | [f32 and bf16](engines/qwen3/) |
| OLMoE-1B-7B-0924 | [bf16](engines/olmoe/) |
| Qwen3-Coder-30B-A3B-Instruct | [MLX 4-bit](engines/qwen3-coder/) |

Use macOS with Metal 4, Apple Silicon, Nix with flakes enabled, and a checkout of
this repository. From its root, download the pinned Apache-2.0 model and build:

```sh
nix develop
python3 engines/smollm2/download.py && forja build smollm2-135m-instruct.metal-apple-m3-ultra.f32
forja run --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --prompt 'The capital of France is' --max-tokens 16
forja --isolated -c engines/smollm2/serve.toml --set "model.model=\"$FORJA_MODELS/SmolLM2-135M-Instruct\"" serve
```

Keep the server running and use a second terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"HuggingFaceTB/SmolLM2-135M-Instruct","messages":[{"role":"user","content":"What is gravity?"}],"max_tokens":32,"temperature":0}'
```

Read [serving and configuration](docs/serving.md),
[adding and optimizing engines](docs/engines.md), [benchmarking](docs/benchmarking.md),
and [architecture and safety](docs/architecture.md). Serving configuration includes
the prefix-cache isolation and stable-weight-file contracts.

Forja is under active development. Supported models have profiles, golden-fixture
verification and recorded benchmarks; context and precision limits are documented
with each engine. Licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
Model weights retain their upstream licenses.
