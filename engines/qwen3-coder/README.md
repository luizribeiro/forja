# Qwen3-Coder

Forja's Qwen3-Coder engine targets Qwen3-Coder-30B-A3B-Instruct with affine four-bit weights and
q8 router gates on Metal for Apple M3 Ultra. Both profiles support a 32768-token context. Prompts
below 2048 tokens use 512-token prefill chunks; longer prompts use 2048-token chunks with exact
remainders capped at 16 tokens. The two planners retain at most two full-chunk and seven exact-tail
graphs between them.

The default profile stores K/V in f16. It passed reduced-precision verification with a 2.01e-4
teacher-forced mean KL, 64/64 top-1 agreement, and a 4.83e-4 maximum layer-relative error; its
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
| f16 | GPU sequential | 1221.81 | 71.53 |
| f16 | GPU pipelined | 1220.90 | 98.02 |
| f32 | GPU sequential | 1218.97 | 71.62 |
| f32 | GPU pipelined | 1220.69 | 98.21 |

Long-context prompt processing benefits more from the smaller cache, but remains outside the
1.25x MLX-LM target. Median wall throughput from matched five-repetition runs was:

| Prompt | f16 | bf16 | f32 | MLX-LM | f16 gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 1255.90 | 1254.56 | 1252.53 | 2961.97 | 2.36x |
| 8192 | 831.47 | 838.09 | 786.85 | 2286.70 | 2.75x |
| 16384 | 525.29 | 526.74 | 485.83 | 1720.25 | 3.28x |

At 32760 prompt tokens, f16 reached 194.70 tok/s with a 36.3 GiB physical-footprint peak and
19.4 GiB guarded RSS, versus 186.61 tok/s, 39.5 GiB, and 22.4 GiB for f32. HTTP measurements for a
2048-token cached prefix plus 33 fresh tokens put f16/f32 uncached TTFT at 1.850/1.853 seconds,
cached TTFT at 135.8/136.5 ms, and cached ITL at 11.76/11.02 ms. The precision ABBA tg128 runs
differed by less than 0.5% after bracketing run-order drift.

The complete confidence intervals and provenance are in the profile's `bench` directory.
