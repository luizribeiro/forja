# Qwen3-Coder

Forja's Qwen3-Coder engine targets Qwen3-Coder-30B-A3B-Instruct with affine four-bit weights and
q8 router gates on Metal for Apple M3 Ultra. The profile supports a 4096-token context and
512-token prefill chunks.

```console
forja build qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja verify --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja bench --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4 --against
```

The port implements Qwen3-Coder GQA, QK normalization, RoPE, sparse expert routing, shared experts,
quantized embeddings and grouped quantized expert projections. Kernel choices are profile data and
the same component supports replay and lazy execution as runtime strategies.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Selection | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| GPU sequential | 1232.33 | 75.11 |
| GPU pipelined | 1232.79 | 98.12 |

The complete confidence intervals and provenance are in the profile's `bench` directory.
