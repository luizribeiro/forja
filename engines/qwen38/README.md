# Qwen3.8-27B

Text-only inference for `mlx-community/Qwen3.8-27B-4bit`: 64 hybrid decoder layers,
affine q4/group-64 weights, BF16 activations, F32 recurrent state and F16 KV.
The Metal profiles target Apple M3 Ultra. The default limits context to 32,768
tokens; `qwen38-27b-4bit.metal-apple-m3-ultra.q4-f32-kv` retains that budget with
F32 KV for diagnostics. Both use q4 weights and BF16 activations.
Prompt processing uses canonical 64-token WY blocks and a final partial block;
decode advances recurrent state one token at a time. Single-lane SDK calls declare
`BatchPhase::Prompt` or `BatchPhase::Decode` explicitly. Prompt submissions start
on a 64-token boundary. Serving supports decode buckets 1/2/4/8 with private
recurrent state and token feedback per lane. Replay retains nine decode graphs,
three prompt shapes (64/512/2,048) and one replaceable tail; bucket 8 runs eagerly.
Serving reuses exact canonical prompt prefixes with a 2 GiB snapshot ceiling.
It saves every 4,096 tokens and the last full 64-token boundary before prompt end,
then replays the entire suffix after the longest hit. The operator can lower
`serving.state_snapshot_bytes`; zero disables retention. After a reply, idle GPU
work recomputes its canonical blocks for exact next-turn token matches. Arriving
requests preempt between blocks; cancellation discards unfinished recompute.
The next turn replays any remaining partial block.

The `qwen38-27b-4bit.metal-apple-m3-ultra.q4-long` profile declares 262,144 tokens
with F16 KV. The operator may lower `serving.max_context`. Prompt scratch remains
bounded by 2,048 tokens. The automatic live quota remains 48 GiB; admission counts
weights, KV backing, state versions, tables, graphs and retained scratch. Long
requests can reduce concurrency or be refused. Profile capacity is not an
accuracy or performance claim.

## Use

Run from the repository root inside `nix develop`:

```sh
python3 engines/qwen38/download.py
forja build qwen38-27b-4bit.metal-apple-m3-ultra.q4
forja run --engine qwen38-27b-4bit.metal-apple-m3-ultra.q4 --prompt 'The capital of France is' --max-tokens 32 --temperature 0
forja --isolated -c engines/qwen38/serve.toml --set "model.model=\"$FORJA_MODELS/Qwen3.8-27B-4bit\"" serve
```

Use the `.q4-f32-kv` or `.q4-long` profile name in the build/run commands for
those variants. When serving, update `model.engine`, `model.profile` and
`model.chat_format` together to select that build.

The downloader verifies upstream revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`
and the original Qwen tokenizer at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
It preserves the original shards and copies text tensor bytes unchanged into
`model.safetensors`; vision tensors are excluded from the guest grant. The text
file hash is `35735f203613e028e6654d7b4f7a3c07da5e5aad896bd5dd453f1529e11d0367`.
The weights and pinned template retain their upstream Apache-2.0 license.

The companion [chat format](../../chat-formats/qwen38/) supports non-thinking text
chat with string content. Tools, images and videos are unsupported.

## Numerical validation

Independent Transformers 5.17.0 CPU fixtures use the same dequantized q4 bytes,
BF16 projection outputs, every layer tap and 32 teacher-forced token choices.
The default F16-KV WebAssembly component passes the reduced-precision end-to-end criteria:

| Prompt | Prompt tokens | Matching choices | Mean KL | Maximum layer relative error |
| --- | ---: | ---: | ---: | ---: |
| Hello | 1 | 31/32 | 0.0007686 | 0.01589 |
| What is 2 + 2? | 8 | 32/32 | 0.0002466 | 0.01518 |
| The capital of France is | 5 | 32/32 | 0.0001519 | 0.01190 |

Operator differentials use normwise relative error at most `1e-5` for F32 and
`1e-2` for BF16/quantized tensors; full-model
checks require mean KL at most `1e-2`, top-1 agreement at least 95% for each prompt,
and per-layer error at most `5e-2`. These are correctness results. No performance is claimed.

The diagnostic F32-KV profile retains q4 weights and BF16 activations. It passes
the same per-prompt gates; the arithmetic fixture also passes with replay enabled.

Long-position operator checks cover 0/63/64, 32K, 128K and 262,143 with integer
RoPE positions, F32/F16 KV and full-width attention. They use independent
references and the same dtype tolerances. Structured recurrent-state checks carry
16 key heads and 48 value heads through 262,144 tokens. Ordinary decay, slow decay
and underflow stay within `1e-5`; near-identity log decay (`-1e-7`) reaches
`1.94e-5` cumulative error against F64, with `1.83e-7` after the first block.
Full-model long-text accuracy remains unqualified.

Real-weight capacity checks reserve the full context under 48 GiB with replay
off/on, count three state versions per lane, and preserve state across refusals.
They populate 64 model tokens; the separate attention checks scan full long KV
caches. These are correctness/capacity diagnostics, not benchmarks.

Regenerate the fixtures with the pinned CPU reference environment:

```sh
uv run --no-project --python 3.12 --with-requirements support/golden/qwen38-requirements.txt python support/golden/qwen38.py --model "$FORJA_MODELS/Qwen3.8-27B-4bit" --out "$FORJA_MODELS/Qwen3.8-27B-4bit/forja-fixtures"
```

## Results

Measured on Apple M3 Ultra (80 GPU cores, 512 GiB), macOS 27.0.1 build 26A434,
using the checkpoint and original tokenizer pinned above. Frozen candidate source
is `c8e62a90`; binary SHA256 starts `a7c0d1f9`, component `002b6cc1` and profile file
`69c58302`. The absolute reference is verified inference source `b7f9fa45`,
component `da16c0b6`: its native measurements use the corrected measurement-only
harness at `b6aa46bd`; HTTP uses its original server binary.

Fresh MLX measurements use mlx-lm 0.31.3 / MLX 0.32.2, identical quantized bytes,
greedy sampling and a 2,048-token prefill step. MLX uses native BF16 KV; Forja uses
F16 KV. Both retain F32 recurrent state. HTTP uses unseeded greedy requests,
decode/prompt concurrency 32/8, ten cache entries and a 24 GiB prefix-cache ceiling
on both servers. Forja includes its 2 GiB state snapshots within that ceiling and
retains a 48 GiB live quota. Multiple live MLX BatchGenerator request IDs were
verified; timing observation was disabled. No competitor measurements were reused.
Serving uses primed 512-token prompts and 128 generated tokens at concurrency 1/4.
Every primary serving request hit the cache: Forja restored 448 tokens and replayed a
64-token canonical suffix; MLX restored 511 and processed one. TTFT includes
restoration and this different suffix work.

Randomized ABBA/BAAB blocks use a fresh loaded native pair with independently
randomized startup order. HTTP starts a fresh server for each whole-workload leg.
Throughput is wall tok/s; TTFT/ITL
are milliseconds. Absolute values are block means with descriptive 95% cluster
bootstrap intervals (4,000 resamples); these intervals have no optional-stopping
coverage claim. Scores are exponentiated median paired log ratios with calibrated
sign-test intervals; latency scores invert the ratio so higher always favors Forja.
Scores can differ from the ratio of the displayed means.
PASS requires an original protected improvement decision at Y=0. Noise/drift
vetoes retain history collection without acceptance. FAIL confirmations cannot
rescue an improvement claim. Intervals are per row, without a joint coverage claim.
Rows with noise/drift vetoes have unprotected score intervals.
Values and intervals use primary runs; FAIL twice records a separate confirmation.
All primary timing runs passed the foreign-GPU audit. Native MLX failures repeated
in valid confirmations. Both serving confirmation attempts had foreign GPU activity,
so those fourteen primary FAILs remain unconfirmed.

### Against the absolute reference

| Workload | Forja [interval] | Reference [interval] | Score [interval] | Primary blocks | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| pp512 | 321.840 [317.524, 326.098] | 300.330 [296.315, 304.341] | 1.149 [0.960, 1.151] | 30 | inconclusive |
| pp2048 | 339.636 [336.852, 342.401] | 325.885 [323.463, 328.285] | 1.042 [1.015, 1.070] | 6 | PASS |
| tg128@512 | 27.383 [27.359, 27.409] | 3.219 [3.215, 3.223] | 8.511 [8.489, 8.522] | 6 | PASS |
| c1 throughput | 22.851 [22.837, 22.864] | 3.161 [3.156, 3.164] | 7.234 [7.208, 7.245] | 6 | PASS |
| c1 TTFT p50 | 961.479 [960.459, 962.481] | 1011.975 [1010.526, 1013.443] | 1.052 [1.049, 1.055] | 30 | history collected |
| c1 TTFT p90 | 961.479 [960.445, 962.461] | 1011.975 [1010.566, 1013.475] | 1.052 [1.049, 1.055] | 30 | history collected |
| c1 TTFT p99 | 961.479 [960.487, 962.457] | 1011.975 [1010.539, 1013.414] | 1.052 [1.049, 1.055] | 30 | history collected |
| c1 ITL p50 | 36.286 [36.255, 36.306] | 310.985 [310.501, 311.563] | 8.565 [8.543, 8.602] | 6 | PASS |
| c1 ITL p90 | 36.663 [36.633, 36.701] | 311.761 [311.326, 312.328] | 8.501 [8.464, 8.536] | 6 | PASS |
| c1 ITL p99 | 37.299 [37.198, 37.420] | 312.284 [312.055, 312.519] | 8.381 [8.356, 8.407] | 30 | history collected |
| c4 throughput | 33.287 [33.269, 33.302] | 11.887 [11.872, 11.900] | 2.800 [2.796, 2.809] | 6 | PASS |
| c4 TTFT p50 | 2412.729 [2410.091, 2415.452] | 2521.800 [2519.195, 2524.552] | 1.046 [1.044, 1.048] | 30 | history collected |
| c4 TTFT p90 | 2518.910 [2516.213, 2521.876] | 2845.204 [2842.632, 2847.737] | 1.130 [1.129, 1.132] | 30 | history collected |
| c4 TTFT p99 | 2518.910 [2516.130, 2521.867] | 2845.204 [2842.665, 2847.825] | 1.130 [1.129, 1.132] | 30 | history collected |
| c4 ITL p50 | 101.942 [101.916, 101.971] | 319.508 [318.976, 320.223] | 3.131 [3.127, 3.149] | 6 | PASS |
| c4 ITL p90 | 102.260 [102.214, 102.312] | 320.340 [319.754, 321.191] | 3.129 [3.125, 3.152] | 6 | PASS |
| c4 ITL p99 | 102.715 [102.674, 102.758] | 320.278 [319.997, 320.584] | 3.117 [3.114, 3.120] | 30 | history collected |

### Against MLX

| Workload | Forja [interval] | Reference [interval] | Score [interval] | Primary blocks | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| pp512 | 318.595 [310.108, 327.111] | 413.661 [402.272, 424.635] | 0.770 [0.704, 0.850] | 6 | FAIL twice |
| pp2048 | 339.732 [336.905, 342.424] | 445.932 [442.899, 448.926] | 0.762 [0.743, 0.781] | 6 | FAIL twice |
| tg128@512 | 27.380 [27.329, 27.418] | 37.246 [37.182, 37.292] | 0.735 [0.732, 0.737] | 6 | FAIL twice |
| c1 throughput | 22.833 [22.818, 22.847] | 32.629 [32.580, 32.676] | 0.699 [0.698, 0.702] | 6 | unconfirmed FAIL |
| c1 TTFT p50 | 962.522 [961.150, 963.995] | 184.588 [183.214, 186.033] | 0.192 [0.189, 0.194] | 30 | unconfirmed FAIL |
| c1 TTFT p90 | 962.522 [961.185, 964.040] | 184.588 [183.228, 185.981] | 0.192 [0.189, 0.194] | 30 | unconfirmed FAIL |
| c1 TTFT p99 | 962.522 [961.133, 964.000] | 184.588 [183.171, 185.995] | 0.192 [0.189, 0.194] | 30 | unconfirmed FAIL |
| c1 ITL p50 | 36.279 [36.256, 36.305] | 29.377 [29.337, 29.422] | 0.809 [0.808, 0.813] | 6 | unconfirmed FAIL |
| c1 ITL p90 | 36.658 [36.638, 36.681] | 29.652 [29.627, 29.678] | 0.809 [0.808, 0.810] | 30 | unconfirmed FAIL |
| c1 ITL p99 | 37.380 [37.290, 37.471] | 30.357 [30.207, 30.488] | 0.812 [0.807, 0.816] | 6 | unconfirmed FAIL |
| c4 throughput | 33.290 [33.278, 33.299] | 93.968 [93.831, 94.107] | 0.354 [0.353, 0.355] | 6 | unconfirmed FAIL |
| c4 TTFT p50 | 2411.811 [2406.864, 2417.902] | 531.869 [524.800, 540.094] | 0.219 [0.217, 0.228] | 6 | unconfirmed FAIL |
| c4 TTFT p90 | 2518.023 [2513.126, 2524.041] | 533.842 [526.714, 542.121] | 0.211 [0.209, 0.219] | 6 | unconfirmed FAIL |
| c4 TTFT p99 | 2518.023 [2513.148, 2524.233] | 533.842 [526.713, 541.807] | 0.211 [0.209, 0.219] | 6 | unconfirmed FAIL |
| c4 ITL p50 | 101.946 [101.931, 101.964] | 38.441 [38.409, 38.474] | 0.377 [0.376, 0.378] | 6 | unconfirmed FAIL |
| c4 ITL p90 | 102.229 [102.207, 102.254] | 38.723 [38.696, 38.752] | 0.379 [0.378, 0.379] | 6 | unconfirmed FAIL |
| c4 ITL p99 | 102.700 [102.674, 102.729] | 39.130 [39.005, 39.270] | 0.381 [0.379, 0.384] | 6 | unconfirmed FAIL |

These records cover a subset. pp16,384, tg128@4,096, the separate
tg128@32,768 group, and all twelve conversation turns (TTFT and decode) remain
unmeasured here. There is no accepted full-profile baseline.

Frozen setup SHA256: `1b7e5c00154c690ddba5308a7cf5082683672db55793b75bac0508d2d2c280bb`.
Summary record SHA256: `e9917e161da257a0aa3219d323e427d1c37269071d71f9a2de725fe6a4ce9504`.
Raw commands, artifacts, benchmark records, gate decisions and confirmations are
retained with these identities. See [benchmarking](../../docs/benchmarking.md) for
the protocol, thresholds, noise checks and interval assumptions.

## Porting notes

Prompt projections use the tiled reduction class; inputs shorter than 16 rows
are zero-padded and cropped. Decode projections use the row-GEMV reduction
class in every serving bucket, with independent rows for batches of 2, 4 and 8.

Decay uses GPU float32 `exp_rounded`/`ln_1p` with a softplus threshold of 20,
matching the reference operation order without scalar readbacks. The block
transform uses F32 MMA products, fixed-tree solves and 32-column carry panels;
16-column panels share its reduction class and produce identical state/output bits. The checkpoint's ordinary RMS weights
already include the unit offset. The
DeltaNet gated norm uses direct multipliers, with normalization rounded to BF16
before the learned weight and float32 SiLU gate. Convolution taps are stored as
`[channels, width, 1]` and are loaded as `[width, channels]`; history keeps raw
pre-convolution rows. Full attention splits query and gate within each head and
rotates only the first 64 of its 256 dimensions. Vision and draft modules are not
executed.
