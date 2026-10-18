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
| f16 | GPU sequential | 1267.45 | 71.19 |
| f16 | GPU pipelined | 1267.26 | 98.14 |
| f32 | GPU sequential | 1218.97 | 71.62 |
| f32 | GPU pipelined | 1220.69 | 98.21 |

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
19.4 GiB guarded RSS, versus 186.61 tok/s, 39.5 GiB, and 22.4 GiB for f32. A warmed 100-request
HTTP run with a 2048-token cached prefix plus 33 fresh tokens measured f16 cached TTFT at
141.3/143.1/144.2 ms and ITL at 11.79/12.40/14.53 ms at p50/p90/p99. Exact cache reuse was 2048
tokens. Old/new tg128 ABBA runs on Qwen3, OLMoE, and Qwen3-Coder differed by less than 0.1% after
bracketing run-order drift.

The complete confidence intervals and provenance are in the profile's `bench` directory.
