# OLMoE

Forja's OLMoE engine targets OLMoE-1B-7B-0924 with bf16 weights and activations on Metal for
Apple M3 Ultra. The profile supports a 4096-token context and 512-token prefill chunks.

```console
forja build olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16
forja verify --engine olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16
forja bench --engine olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16 --against
```

The port implements grouped-query attention, rotary embeddings, sparse expert routing and grouped
expert projections. Addition, SiLU multiplication and router softmax are built by the guest, while
the profile carries the verified attention, dense projection and MoE kernel picks. Q/K RMS
normalization and RoPE retain their host paths. Substituting either one independently changed f32
rounding enough to flip a valid near-tie expert route and raise the maximum final-hidden-state
relative error to about 0.056, above the 0.05 acceptance limit. The generic guest Row pipeline can
use a different threadgroup reduction grouping, while guest RoPE computes inverse frequencies and precise
trigonometric functions in the program instead of consuming the host kernel's precomputed f32
frequencies. This is not an expressiveness limit of guest programs: a future builder can expose the
same reduction schedule and precomputed frequency inputs, but the current builders do not promise
bitwise parity with those host kernels.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra. The TG
column is the context-9 result; the accepted record also covers contexts 512 and 2048.

| Selection | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| GPU sequential | 916.24 | 84.89 |
| GPU pipelined | 916.84 | 98.49 |

The paged-cache profile was screened with a five-repetition A/B/B/A against the otherwise
identical contiguous profile. Pipelined wall/GPU deltas at pp512 and tg128 contexts 9/512/2048
were +0.13/+0.06%, +2.47/+2.38%, -0.17/-0.30%, and -0.65/-0.64%. The context-2048 result matches
the accepted long-context cost in exchange for batching and multi-entry prefix reuse.

The complete confidence intervals and provenance are in the profile's `bench` directory.
