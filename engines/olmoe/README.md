# OLMoE

Forja's OLMoE engine targets OLMoE-1B-7B-0924 with bf16 weights and activations on Metal for
Apple M3 Ultra. The profile supports a 4096-token context and 512-token prefill chunks.

```console
forja build olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16
forja verify --engine olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16 --each-tuning
forja bench --engine olmoe-1b-7b-0924.metal-apple-m3-ultra.bf16 --against
```

The port implements grouped-query attention, rotary embeddings, sparse expert routing and grouped
expert projections. Its profile carries the verified attention, dense projection and MoE kernel
picks used by the single runtime-selectable component.

Accepted pp512/tg128 results use 30 measured repetitions on an 80-core Apple M3 Ultra:

| Selection | PP tok/s | TG tok/s |
| --- | ---: | ---: |
| GPU sequential | 902.07 | 82.83 |
| GPU pipelined | 898.60 | 95.31 |

The complete confidence intervals and provenance are in the profile's `bench` directory.
