# Qwen3-Coder

Forja's Qwen3-Coder engine targets Qwen3-Coder-30B-A3B-Instruct with affine four-bit weights and
q8 router gates on Metal for Apple M3 Ultra. Both profiles support a 32768-token context. Prompts
below 2048 tokens use 512-token prefill chunks; longer prompts use 2048-token chunks with exact
remainders capped at 16 tokens. The two planners retain at most two full-chunk and seven exact-tail
graphs between them.

The default profile stores K/V in f16. It passed reduced-precision verification with a 2.012e-4
teacher-forced mean KL, 64/64 top-1 agreement, and a 4.826e-4 maximum layer-relative error; its
64/64 free-running agreement is informational. The `.kv-f32` profile keeps exact f32 cache storage
and passed strict verification with exact teacher-forced and free-running top-1 agreement and a
4.18e-6 maximum layer-relative error. Choose `.kv-f32` when exact-cache reproducibility matters;
otherwise use the default f16 profile for its lower memory use and better long-context throughput.

```console
forja build qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja verify --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4
forja bench --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4 --against

forja build qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4.kv-f32
forja verify --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4.kv-f32
forja bench --engine qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4.kv-f32 --against
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

| KV cache | Selection | PP tok/s | TG tok/s |
| --- | --- | ---: | ---: |
| f16 | GPU sequential | 1308.62 | 82.26 |
| f16 | GPU pipelined | 1299.95 | 105.31 |
| f32 | GPU sequential | 1266.44 | 82.78 |
| f32 | GPU pipelined | 1246.92 | 104.88 |

Full-model batched decode uses alternating 512- and 2048-token cached contexts, 128 timed decode
steps, three warmups and 30 measured repetitions. The aggregate throughput and each lane's ITL are:

| Bucket | Execution | Wall tok/s (95% CI) | GPU tok/s (95% CI) | Wall ITL ms (95% CI) | GPU ITL ms (95% CI) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | graph | 75.29 (73.97–76.30) | 96.07 (95.38–97.06) | 13.28 (13.11–13.52) | 10.41 (10.30–10.48) |
| 2 | graph | 108.85 (107.65–109.68) | 138.69 (136.92–139.52) | 18.37 (18.23–18.58) | 14.42 (14.34–14.61) |
| 4 | graph | 145.32 (144.84–145.91) | 182.57 (181.88–183.06) | 27.53 (27.41–27.62) | 21.91 (21.85–21.99) |
| 8 | eager | 149.38 (149.26–149.46) | 215.79 (215.41–216.14) | 53.56 (53.53–53.60) | 37.07 (37.01–37.14) |

The eager bucket 8 gains only 2.8% aggregate wall throughput over bucket 4 while its per-lane wall
ITL is 94.6% higher. Against bucket 2, its wall ITL is 191.5% higher. The record was captured by
`forja bench` at commit `abe0761`, with engine SHA-256 `7a96b390d54b200ea050feb9c5b31244701e275063f85902d303f618b4fc01cd`,
model revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`, and an 80-core Apple M3 Ultra on macOS 27.0.1.
Full confidence intervals and provenance are in `bench/qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4/batched-decode.json`.

Continuous HTTP serving uses at most eight active lanes and caps each prompt chunk at the profile's
512-token serving budget. With 30 measured repetitions after three warmups, the 33-token greedy
workload produced:

| Server | c1 output tok/s | c2 | c4 | c8 | c4/c1 | c1/c4 streaming p99 ITL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Forja | 104.20 | 151.33 | 205.23 | 163.51 | 1.97x | 9.99 / 19.06 ms |
| mlx_lm 0.31.3 | 91.51 | 152.36 | 235.05 | 337.69 | 2.57x | 10.94 / 15.19 ms |

Forja passes the concurrency gate: streaming concurrency-four aggregate decode is 1.99x its
concurrency-one result, while p99 ITL is 1.91x. It completed the 34-case, 3,360-request suite with
no errors and 150 requested stream cancellations, and all 21 comparable serial-versus-concurrent
outputs matched. At concurrency four, Forja served 1.603 requests/s and 52.91 prompt tok/s;
streaming measured 2.315 requests/s, 94.93 prompt tok/s, 203.76 output tok/s, 137.56 ms p99 TTFT,
and 1.734 s p99 end-to-end latency. The corresponding mlx_lm streaming result was 2.551
requests/s, 104.58 prompt tok/s, 218.08 output tok/s, 716.09 ms p99 TTFT, and 2.263 s p99
end-to-end latency.

mlx_lm ran with `--decode-concurrency 32 --prompt-concurrency 8`. Its greedy throughput scales
with client concurrency, confirming that its batched path ran, but only 13 of 21 concurrent outputs
matched their serial references. Seeded sampling is not batchable on mlx_lm: its c1/c4 throughput
was 98.17/98.21 output tok/s. Forja's seeded c1/c4 throughput was 103.55/203.89 output tok/s and
the concurrent output matched serial.

The representative Forja concurrency-four case reported a 3,271,559,200-byte pool, a 384-byte
table, zero pool-attributed graph and scratch bytes, 17.60 GiB RSS, 71.93 GPU seconds, 74.84
scheduler seconds, and 4.18 queue seconds. mlx_lm reported 16.49 GiB RSS; its server does not expose
the corresponding device-memory, GPU-time, or host-phase counters. Per-case request, token,
TTFT/ITL/end-to-end percentile, cache, memory, GPU, host-phase, error, cancellation, configuration,
and provenance data are in `serving-concurrency.json` and `serving-concurrency-mlx.json` beside the
other accepted records.

The f16 profile uses half-precision Q, K, V, and probability MMA inputs with f32 score, softmax,
and output accumulation. Median wall throughput from matched five-repetition runs was:

| Prompt | f16 before | f16 after | Improvement | f32 | MLX-LM | f16 gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1254.54 | 1362.10 | 8.6% | 1251.42 | 2961.97 | 2.17x |
| 8192 | 833.26 | 1144.98 | 37.4% | 786.49 | 2286.70 | 2.00x |
| 16384 | 524.50 | 902.93 | 72.2% | 485.89 | 1720.25 | 1.91x |

At 16K, the attention profile fell from 5.237 to 1.835 seconds, a 2.85x kernel-level speedup,
and its GPU share fell from 77.8% to 58.9%. The end-to-end result remains outside the 1.25x
MLX-LM target; quantized gather matmul is now the largest secondary cost. The f32 results remain
within 0.1% of the preceding measurements because that profile retains the original Steel path.
The f16 verification rerun changed teacher-forced mean KL only from 2.01e-4 to 2.012e-4 and kept
64/64 top-1 agreement.

At 32760 prompt tokens, f16 reached 194.70 tok/s with a 36.3 GiB physical-footprint peak and
19.4 GiB guarded RSS, versus 186.61 tok/s, 39.5 GiB, and 22.4 GiB for f32. Eviction-faithful
100-request HTTP A/B/B/A runs with a 2048-token cached prefix plus 33 fresh tokens measured
contiguous f16 cached TTFT medians of 139.71 and 140.67 ms and paged medians of 153.81 and
155.60 ms. Both paged 95% confidence intervals overlapped both contiguous intervals. Paged ITL
medians were 10.64 and 10.57 ms, versus 10.80 ms for both contiguous legs, and exact cache reuse
was 2048 tokens. Old/new tg128 ABBA runs on Qwen3, OLMoE, and Qwen3-Coder differed by less than
0.1% after bracketing run-order drift.

The complete confidence intervals and provenance are in the profile's `bench` directory.

## Paged serving memory ceiling

The f16 paged profile stores 6 MiB per 64-token page across its 96 cache planes. The table below
charges eight admitted lanes, two slot banks, per-sequence tables, generation metadata, and a
3 MiB conservative prefix-index allowance. The fixed working set is 20,164,345,856 bytes: the
17,181,071,994-byte weights plus ordinary tensors, two replay arenas, and measured scratch
high-water storage. Page references are not charged a second time.

For `P = context / 64` pages, the metadata column is the 3 MiB prefix-index allowance plus
`2 × (8 × P × 8 + 8 × 16)` bytes for the two charged slot banks, `2 × 8 × P × 8` for their GPU
table buffers, `8 × P × 8` for active sequence tables, and `P × 4` for generations. This totals
3,166,720, 3,228,928, and 3,311,872 bytes for the three rows respectively.

| Context ceiling | Cache slabs | Slot/table metadata | Fixed working set | Charged ceiling | 48 GiB headroom |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 3.00 GiB | < 3.1 MiB | 18.78 GiB | 21.78 GiB | 26.22 GiB |
| 16,384 | 12.00 GiB | < 3.1 MiB | 18.78 GiB | 30.78 GiB | 17.22 GiB |
| 32,768 | 24.00 GiB | < 3.2 MiB | 18.78 GiB | 42.78 GiB | 5.22 GiB |

The slab allocator grows in 3 GiB units because one slab spans a 32K f16 sequence. Eight 4K lanes
therefore share one slab, eight 16K lanes require four, and eight 32K lanes require eight. The
32K row is an accounting ceiling, not a concurrency target; admission may use the remaining
headroom only when the complete reservation fits the unchanged quota.
