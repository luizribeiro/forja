# Qwen3

Forja's dense Qwen3 engine currently targets Qwen3-0.6B on Metal for Apple M3 Ultra. The engine
supports the model's native 40960-token maximum. The shipped f32 and bf16 profiles declare a
4096-token context and use 512-token prefill chunks.

```console
forja build qwen3-0.6b.metal-apple-m3-ultra.bf16
forja verify --engine qwen3-0.6b.metal-apple-m3-ultra.bf16
forja bench --engine qwen3-0.6b.metal-apple-m3-ultra.bf16 --against
```

The port uses Qwen3 GQA, QK normalization, RoPE, gated MLP blocks, retained KV caches, chunked
prefill and guest-side graph replay. The guest builds scalar elementwise and row programs while
the profiles retain shape-specific Metal algorithm picks.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra. The TG
column is the context-9 result; the accepted records also cover contexts 512 and 2048.

| Profile | Selection | PP tok/s | TG tok/s |
| --- | --- | ---: | ---: |
| f32 | GPU sequential | 4559.34 | 101.77 |
| f32 | GPU pipelined | 4475.18 | 125.61 |
| bf16 | GPU sequential | 5978.33 | 114.28 |
| bf16 | GPU pipelined | 5967.28 | 149.68 |

The paged-cache profiles were screened with a five-repetition A/B/B/A against otherwise identical
contiguous profiles. Pipelined wall/GPU deltas at pp512 and tg128 contexts 9/512/2048 were
f32: +1.21/+0.85%, +3.66/+4.09%, -0.69/-0.59%, -1.14/-1.16%; and bf16:
+7.21/+6.49%, +5.14/+5.10%, +2.21/+2.53%, +0.86/+0.86%. The f32 context-2048 cost is accepted:
f32 is the precision-reference profile, while bf16 is the speed profile and improves by up to 7%.

The complete confidence intervals and provenance are in each profile's `bench` directory.
