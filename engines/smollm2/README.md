# SmolLM2-135M-Instruct

This is the worked example for [adding and shipping an engine](../../docs/engines.md).
Both implementations build into the same component. The reference profile selects
[`reference.rs`](src/reference.rs); the optimized f32 profile selects
[`optimized.rs`](src/optimized.rs) through the registered `optimized` tuning at load
time. No cargo feature removes either path.

The upstream [config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json)
declares Llama with 30 layers, hidden width 576, intermediate width 1536, nine query
heads, three KV heads and head width 64. RMS norm uses epsilon 1e-5; rotary embeddings
pair half-heads with theta 100000. The final vocabulary projection shares the
49152-row embedding table. Native context is 8192 tokens. The model is
[Apache-2.0 licensed](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md).
[`download.py`](download.py) pins that revision and verifies every downloaded file.
Both profiles promote the checkpoint's bf16 weights to f32 and use f32 activations
and KV storage; the optimized profile uses 512-token prefill chunks.

```sh
python3 engines/smollm2/download.py
forja build smollm2-135m-instruct.metal-apple-m3-ultra.reference
forja build smollm2-135m-instruct.metal-apple-m3-ultra.f32
forja verify --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --fixtures "$FORJA_MODELS/golden/smollm2-135m-instruct"
forja bench --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --against
```

Run these inside `nix develop`; the root [quickstart](../../README.md) includes
serving. Generate independent fixtures first using
[`support/golden/smollm2.py`](../../support/golden/smollm2.py).

The reference file follows Llama's execution order: `embed`, `decoder_layer`,
`attention`, `mlp`, `rms_norm`, then `forward`'s tied projection. It uses a contiguous
KV cache, fixed portable matrix multiplication and decomposed causal attention.
[`lib.rs`](src/lib.rs) shares validated dimensions, tensor loading and component
glue. Its `Config::tiny()` has the same GQA, RoPE, SwiGLU and tied-output structure
with random weights for [fast differential tests](../../crates/forja-host/tests/smollm2.rs).
Those tests compare both logits and every layer tap using the shared normwise f32
tolerance. Full profiles additionally pass independent transformer-f32 verification
on four prompts and 32 greedy steps per prompt.

Optimization follows the execution cost: choose matrix and attention variants in
[`variants.rs`](src/variants.rs); fuse gated projection epilogues and residual RMS
norm in [`optimized.rs`](src/optimized.rs); use host-owned paged KV; capture decode
in [`replay.rs`](src/replay.rs); then reuse bounded prefill chunks and exact tails.
`fused-epilogues` and `chunked-prefill` are separately selectable tunings;
`engine.replay=false` disables capture. Prompt boundaries stay fixed:
`prefill_chunk_multiple` is `None`, so this engine makes no bitwise claim about
merging serving chunks. The model verification tests also exercise two full
prefill chunks, repeated exact tails, page crossings and retained-token decode.

Measured optimization results and the `mlx_lm` comparison are recorded under
[`bench/`](bench/). Use [benchmarking](../../docs/benchmarking.md) for measurement
modes, confidence intervals, provenance and acceptance rules.
