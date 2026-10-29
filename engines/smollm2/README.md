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
forja -c bench/suites/default.toml bench --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --against
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
`Config::tiny_cpu()` runs both paths in the non-GPU commit hook; Metal cases run
in the serialized push-stage group. Those tests compare both logits and every layer tap using the shared normwise f32
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

The optimization sequence has five independently inspectable changes:

1. Pick row-one transposed GEMV and prefill Steel matrix kernels through
   [`Variants`](src/variants.rs), with separate prefill and decode attention sites.
2. Fuse the up projection with SiLU gating; fuse residual addition with the next
   RMS norm. `fused-epilogues` controls this tradeoff in [`optimized.rs`](src/optimized.rs).
3. Replace per-layer contiguous caches with the declared paged planes. The
   [paged profile](profiles/smollm2-135m-instruct.metal-apple-m3-ultra.f32.toml)
   also enables serving prefix reuse; the [contiguous control](bench/contiguous/contiguous.toml)
   retains the same model and precision.
4. Capture single-token forward graphs with a position parameter in
   [`replay.rs`](src/replay.rs). Shared glue stages feedback and performs sampling.
5. Enable `chunked-prefill` to reuse full 512-token prompt graphs and bounded exact
   tails. Capture is lazy; repeated tail sizes enter the SDK's bounded graph cache.

The table below uses **five-repetition descriptive screens**, one warmup,
pp512/tg128, pipelined selection, and context 9 for TG. Each delta is against the
preceding row. Small differences and overlapping intervals are not improvement
PASS claims. Background CPU warnings remain in the records.


| Change and record | PP wall tok/s | TG wall tok/s | PP delta | TG delta |
| --- | ---: | ---: | ---: | ---: |
| [Reference](bench/smollm2-135m-instruct.metal-apple-m3-ultra.reference/screen-reference.json) | 8422.72 | 51.21 | — | — |
| [Variant picks](bench/contiguous/screen-variants.json) | 8526.83 | 110.00 | +1.24% | +114.81% |
| [Fused epilogues](bench/contiguous/screen-fusion.json) | 7965.36 | 110.95 | -6.58% | +0.87% |
| [Paged cache](bench/smollm2-135m-instruct.metal-apple-m3-ultra.f32/screen-paging.json) | 8000.60 | 114.94 | +0.44% | +3.60% |
| [Decode replay](bench/smollm2-135m-instruct.metal-apple-m3-ultra.f32/screen-replay.json) | 7997.62 | 139.18 | -0.04% | +21.09% |
| [Chunked prefill](bench/smollm2-135m-instruct.metal-apple-m3-ultra.f32/screen-chunks.json) | 8613.49 | 137.62 | +7.70% | -1.12% |

Fusion's prefill cost is visible; paging's small decode gain and chunking's small
decode cost need stronger evidence before treating them as general effects. The
profiles make those choices inspectable and reversible.

The **accepted absolute baselines** use 30 repetitions and three warmups, with
both sequential and pipelined selections and contexts 9/512/2048/4000. Below are
the pipelined wall medians with the records' 95% intervals. Measurements used the
frozen `d91117f` build on an 80-core Apple M3 Ultra, macOS 27.0.1, on 2026-10-29;
the records preserve exact binary, component, profile, configuration and model hashes.
The later native CPU-test compatibility change preserves the Metal instruction
stream; accepted records retain their original measured artifact identities.

| Implementation and record | PP512 wall tok/s | TG128 at context 9 wall tok/s |
| --- | ---: | ---: |
| [Reference f32](bench/smollm2-135m-instruct.metal-apple-m3-ultra.reference/accepted.json) | 8572.83 (8554.59–8605.81) | 51.27 (51.20–51.40) |
| [Optimized f32](bench/smollm2-135m-instruct.metal-apple-m3-ultra.f32/accepted.json) | 8614.97 (8568.43–8633.71) | 137.53 (136.69–137.84) |
| [MLX-LM bf16](bench/mlx-bf16/record.json) | 25521.05 (25279.01–25644.93) | 479.47 (478.93–480.52) |

MLX-LM 0.31.3 with MLX 0.32.2 is faster in both rows. It uses native bf16 while
Forja uses f32-promoted weights, activations and KV. Both use the same pinned
checkpoint, synthetic-token workload, greedy selection and
[suite](bench/mlx-bf16/suite.toml); the MLX driver disables EOS and records its
pipelined generation path. This is an explicitly different-precision comparison.
The small model's remaining decode overhead includes feedback staging and sampling
outside the forward capture; further tuning belongs in the guest.

Use [benchmarking](../../docs/benchmarking.md) for the default fast gate,
exhaustive fixed repetitions, confidence intervals, provenance and acceptance.
There is no calibrated sequential PASS claim for these first absolute baselines.
