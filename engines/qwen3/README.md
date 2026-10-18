# Qwen3

Forja's dense Qwen3 engine currently targets Qwen3-0.6B on Metal for Apple M3 Ultra. Profiles are
provided for f32 and bf16 activations and weights, with a 4096-token context and 512-token prefill
chunks.

```console
forja build qwen3-0.6b.metal-apple-m3-ultra.bf16
forja verify --engine qwen3-0.6b.metal-apple-m3-ultra.bf16
forja bench --engine qwen3-0.6b.metal-apple-m3-ultra.bf16 --against
```

The port uses Qwen3 GQA, QK normalization, RoPE, gated MLP blocks, retained KV caches, chunked
prefill and guest-side graph replay. The guest builds scalar elementwise and row programs while
the profiles retain shape-specific Metal algorithm picks.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Profile | Selection | PP tok/s | TG tok/s |
| --- | --- | ---: | ---: |
| f32 | GPU sequential | 4460.75 | 93.03 |
| f32 | GPU pipelined | 4446.63 | 119.47 |
| bf16 | GPU sequential | 5562.97 | 103.82 |
| bf16 | GPU pipelined | 5565.20 | 140.72 |

The complete confidence intervals and provenance are in each profile's `bench` directory.
