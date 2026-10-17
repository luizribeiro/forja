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

## API notes

The chat format accepts OpenAI function tools and `parallel_tool_calls` set to either value, but
formats at most one tool call per assistant turn. Clients can append its tool result and continue
the conversation for the next call.

The profiled decode graph eagerly compiles two epilogue libraries and three pipeline
specializations. The prefill graph compiles one library and one specialization. Replay requires
zero compile fallbacks in both modes.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Selection | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| GPU sequential | 1227.60 | 71.51 |
| GPU pipelined | 1221.48 | 98.05 |

The complete confidence intervals and provenance are in the profile's `bench` directory.
