# Qwen3

Forja's dense Qwen3 engine currently targets Qwen3-0.6B on Metal for Apple M3 Ultra. Profiles are
provided for f32 and bf16 activations and weights, with a 4096-token context and 512-token prefill
chunks.

```console
forja build qwen3-0.6b.metal-apple-m3-ultra.bf16
forja verify --engine qwen3-0.6b.metal-apple-m3-ultra.bf16 --each-tuning
forja bench --engine qwen3-0.6b.metal-apple-m3-ultra.bf16 --against
```

The port uses Qwen3 GQA, QK normalization, RoPE, gated MLP blocks, retained KV caches, chunked
prefill and guest-side graph replay. The profiles select fused residual norm, QK-norm/RoPE and
SiLU-multiply slots (f32 also keeps the final-norm fusion; bf16 dropped it after measuring it
neutral) plus shape-specific Metal algorithm picks.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Profile | Selection | PP tok/s | TG tok/s |
| --- | --- | ---: | ---: |
| f32 | GPU sequential | 4443.06 | 98.63 |
| f32 | GPU pipelined | 4428.78 | 120.63 |
| bf16 | GPU sequential | 5520.28 | 110.17 |
| bf16 | GPU pipelined | 5539.65 | 141.23 |

The complete confidence intervals and provenance are in each profile's `bench` directory.
