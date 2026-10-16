# Qwen3-Coder

Forja's Qwen3-Coder engine targets Qwen3-Coder-30B-A3B-Instruct with affine four-bit weights and
q8 router gates on Metal for Apple M3 Ultra. The profile supports a 32768-token context and
512-token prefill chunks.

```console
forja build qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja verify --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja bench --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4 --against
```

The port implements Qwen3-Coder GQA, QK normalization, RoPE, sparse expert routing, shared experts,
quantized embeddings and grouped quantized expert projections. Kernel choices are profile data and
the same component supports replay and lazy execution as runtime strategies.

Q/K RMS normalization and RoPE run as one guest Row program in this engine. Its different reduction
grouping and program-computed precise trigonometry remain within the quantized model's verification
tolerance: Python and Rust layer checks pass and all 32 teacher-forced and free-running tokens match.
The trusted RMSNorm and RoPE operations remain available for OLMoE, whose near-tie routing is
sensitive to those f32 rounding differences; the Coder does not depend on the trusted variants.

The profiled decode graph eagerly compiles two epilogue libraries and three pipeline
specializations. The prefill graph compiles one library and one specialization. Replay requires
zero compile fallbacks in both modes.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Selection | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| GPU sequential | 1228.85 | 71.09 |
| GPU pipelined | 1225.49 | 97.91 |

The complete confidence intervals and provenance are in the profile's `bench` directory.
