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
four prompt shapes (64/128/512/2,048) and one replaceable tail. With a recording
quota of at least 8,192 dispatches it also declares and captures two verify
shapes (one or two drafts); lower quotas serve plain decode. Bucket 8 supports
combined decode and runs eagerly. Verification
steps convolution, recurrent state, attention and sampling one row at a time with
decode arithmetic. The standard q4 profile uses host prompt lookup with an
eight-token minimum match and a maximum suffix of 32 tokens. The serving example
explicitly enables draft mode with
`limits.dispatches_per_list = 8192`; the host default stays at 4,096.
Serving reuses exact canonical prompt prefixes with a 2 GiB snapshot ceiling.
It saves every 4,096 tokens and the last full 64-token boundary before prompt end,
then replays the entire suffix after the longest hit. The operator can lower
`serving.state_snapshot_bytes`; zero disables retention. After a reply, idle GPU
work recomputes its canonical blocks for exact next-turn token matches. Arriving
requests preempt between blocks; cancellation discards unfinished recompute.
The next turn replays any remaining partial block.

The `qwen38-27b-4bit.metal-apple-m3-ultra.q4-long` profile declares 262,144 tokens
with F16 KV. The operator may lower `serving.max_context`. Prompt scratch remains
bounded by 2,048 tokens. With `limits.enforce_gpu_time = true`, profile
`prompt-chunks` bounds actual pieces to 512
through 32,768 context tokens and 128 afterward. The standard q4 profile uses
512 through its 32,768-token limit. The F32 KV profile uses 512 through 2,048
and 128 afterward. These guest declarations keep each submission within the
host's cumulative forecast budget. The automatic live quota remains 48 GiB; admission counts
weights, KV backing, state versions, tables, graphs and retained scratch. Long
requests can reduce concurrency or be refused. Profile capacity is not an
accuracy or performance claim.

## Use

Run from the repository root inside `nix develop`:

```sh
python3 engines/qwen38/download.py
forja build qwen38-27b-4bit.metal-apple-m3-ultra.q4
forja --set limits.dispatches_per_list=8192 run --engine qwen38-27b-4bit.metal-apple-m3-ultra.q4 --prompt 'The capital of France is' --max-tokens 32 --temperature 0
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

## Prompt lookup

Host prompt lookup is enabled by default for profiles declaring a lookup policy.
It proposes tokens from matching suffixes in the request's accepted history and
verifies them with the target model before committing them. The default q4
profile matches 8–32 tokens and proposes up to two tokens per round. It engages
only for one active, unconstrained request when Verify fits the host limits,
within the profile's context cap; drafting is disabled at 32K and beyond.
Set `--set serving.prompt_lookup=false` on `forja serve` to opt out.
`forja bench completion --mode plain` keeps plain decoding explicit.

Regular measurements compare lookup with plain decoding in the same frozen
binary SHA256 `887da03e8741…` at source `3fd4bc49`, component `38b94d70f271…`,
profile file `2ad8e5893e2d…`, using model revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818` on Apple M3 Ultra (80 GPU cores,
512 GiB), macOS 27.0.1 build 26A434.
All 17 rows ran 30 randomized ABBA/BAAB blocks with a fresh worker pair per
block and randomized startup order; all 1,020 stream checks matched. The table
shows decode-wall speed ratios and rowwise 95% calibrated sign-test intervals.

| Workload | Lookup/plain ratio [95% CI] |
| --- | ---: |
| Code edit, short | 1.214 [1.212, 1.215] |
| Code edit, 4K | 1.210 [1.209, 1.211] |
| Greedy JSONL copy, short | 1.207 [1.206, 1.209] |
| Greedy JSONL copy, 4K | 1.203 [1.202, 1.204] |
| Seeded JSONL copy | 1.144 [1.144, 1.146] |

The release decision accepts these results with a 1.14× bar for seeded copy.
Ordinary, concurrent and prompt rows meet the 0.98× bar except short ordinary
code: 0.9820 [0.9797, 0.9833], inconclusive against that bar and accepted as a known
caveat. The original reports retain their automated `history-collected` verdicts
where applicable; verified noise/drift protection was unavailable. Completion
benchmarks have no accepted-record importer, so the original records, wall/GPU
observations and full artifact identities remain at
`~/docs/forja/evidence/e/gate/` (`setup.json`, `summary.json`, `regular/`).
No 32K measurements are included.

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
off/on, count three state versions per lane plus one shared verify spare, and
preserve state across refusals.
They populate 64 model tokens; the separate attention checks scan full long KV
caches. These are correctness/capacity diagnostics, not benchmarks.

Regenerate the fixtures with the pinned CPU reference environment:

```sh
uv run --no-project --python 3.12 --with-requirements support/golden/qwen38-requirements.txt python support/golden/qwen38.py --model "$FORJA_MODELS/Qwen3.8-27B-4bit" --out "$FORJA_MODELS/Qwen3.8-27B-4bit/forja-fixtures"
```

## Results

Historical measurements on Apple M3 Ultra (80 GPU cores, 512 GiB), macOS 27.0.1
build 26A434, using the pinned checkpoint and original tokenizer above. Candidate
source is `3bd3f609`; binary SHA256 starts `4fc0f98a4567`, component
`4d2c44b8ed0d`, default profile file `3628daaf0fa4`.
The absolute reference is verified inference source `b7f9fa45`, component
`da16c0b6`; native measurements use its corrected measurement-only harness at
`b6aa46bd`, while HTTP uses its original server. The same frozen `b6aa46bd`
HTTP client drives every server.

**These results apply only to `3bd3f609`.** Source `8e4f0bc9` includes two host
kernel safety fixes that change the host binary. These records do not validate
that binary; completing any pending coverage requires a newly frozen candidate
and fresh measurements, including fresh primary runs for unconfirmed failures.

Fresh competitor measurements use mlx-lm 0.31.3 / MLX 0.32.2, identical affine
q4/group-64 tensors, greedy sampling and a 2,048-token prefill step. MLX uses
BF16 KV; Forja uses F16 KV. Both retain F32 recurrent state. HTTP omits seed and
uses decode/prompt concurrency 32/8, ten cache entries and a 24 GiB prefix-cache
ceiling. Forja includes its 2 GiB state snapshots within that ceiling and retains
a 48 GiB live quota. The multiple-request MLX BatchGenerator admission proof was
revalidated against installed runtime/adapter hashes; observation was disabled
for timing. No competitor measurements were reused.

Serving uses primed 512-token prompts and 128 generated tokens. The measured
c1/c4 requests all hit the cache: Forja restored 448 tokens, MLX 511. TTFT includes
restoration and this different suffix work. Serial-versus-c4 output tokens matched
within each implementation; these workloads had no request errors.

Randomized ABBA/BAAB blocks use a fresh loaded native pair per block, randomized
startup order and fresh request state. HTTP starts a fresh server per workload leg.
Throughput is wall tok/s; TTFT/ITL are milliseconds. Absolute values are block
means with descriptive 95% cluster bootstrap intervals (4,000 resamples), without
optional-stopping coverage. Scores are exponentiated median paired log ratios
with calibrated sign-test intervals; latency scores invert the ratio so higher
favors Forja. Intervals are per row, with no joint coverage claim.
PASS requires original protected improvement at Y=0; X=2% is recorded in the plans.
Noise/drift vetoes require all 30 blocks and prevent acceptance of a positive result.
Their score intervals are unprotected; cap means inconclusive. FAIL twice requires
one fresh confirmation passing the foreign-GPU audit. Confirmation cannot rescue
an improvement claim. Values use valid primary runs. The invalid first absolute-
reference serving attempt is excluded; its single retry passed the GPU audit.

### Against the absolute reference

| Workload | Forja [interval] | Reference [interval] | Score [interval] | Blocks | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| pp512 | 319.343 [315.857, 322.762] | 307.926 [304.766, 311.117] | 0.980 [0.979, 1.127] | 30 | inconclusive |
| pp2048 | 339.908 [337.806, 342.015] | 327.662 [325.839, 329.477] | 1.024 [1.022, 1.064] | 6 | PASS |
| tg128@512 | 28.420 [28.412, 28.427] | 3.220 [3.210, 3.229] | 8.826 [8.776, 8.881] | 6 | PASS |
| c1 throughput | 26.716 [26.683, 26.750] | 3.157 [3.154, 3.160] | 8.456 [8.446, 8.485] | 6 | PASS |
| c1 TTFT p50 | 324.828 [322.243, 327.284] | 1008.346 [1005.275, 1011.114] | 3.104 [3.059, 3.148] | 6 | PASS |
| c1 TTFT p90 | 324.828 [322.221, 327.284] | 1008.346 [1005.344, 1011.247] | 3.104 [3.059, 3.148] | 6 | PASS |
| c1 TTFT p99 | 324.828 [322.233, 327.284] | 1008.346 [1005.376, 1011.247] | 3.104 [3.059, 3.148] | 6 | PASS |
| c1 ITL p50 | 34.938 [34.913, 34.969] | 311.332 [311.069, 311.564] | 8.910 [8.893, 8.926] | 6 | PASS |
| c1 ITL p90 | 35.313 [35.250, 35.409] | 312.652 [312.267, 312.993] | 8.859 [8.816, 8.874] | 6 | PASS |
| c1 ITL p99 | 36.180 [36.080, 36.280] | 313.464 [313.061, 313.867] | 8.679 [8.600, 8.742] | 6 | PASS |
| c4 throughput | 49.431 [49.401, 49.456] | 11.892 [11.885, 11.899] | 4.158 [4.146, 4.162] | 6 | PASS |
| c4 TTFT p50 | 1179.471 [1137.043, 1216.730] | 2515.304 [2513.343, 2517.302] | 1.987 [1.978, 2.268] | 30 | inconclusive: history collected |
| c4 TTFT p90 | 1352.244 [1346.843, 1359.684] | 2835.419 [2832.088, 2838.902] | 2.100 [2.055, 2.109] | 6 | PASS |
| c4 TTFT p99 | 1352.244 [1346.672, 1360.086] | 2835.419 [2831.908, 2838.590] | 2.100 [2.055, 2.109] | 6 | PASS |
| c4 ITL p50 | 71.254 [71.250, 71.258] | 319.312 [319.130, 319.501] | 4.480 [4.477, 4.484] | 30 | inconclusive: history collected |
| c4 ITL p90 | 71.441 [71.436, 71.447] | 320.275 [320.069, 320.503] | 4.481 [4.478, 4.486] | 30 | inconclusive: history collected |
| c4 ITL p99 | 79.524 [78.895, 80.139] | 321.455 [321.207, 321.705] | 3.985 [3.965, 4.085] | 30 | inconclusive: history collected |

### Against MLX

| Workload | Forja [interval] | Reference [interval] | Score [interval] | Blocks | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| pp512 | 321.174 [313.584, 328.459] | 416.639 [406.099, 426.048] | 0.771 [0.712, 0.845] | 6 | FAIL twice |
| pp2048 | 341.939 [339.888, 343.983] | 445.076 [442.222, 447.953] | 0.778 [0.748, 0.782] | 6 | FAIL twice |
| tg128@512 | 28.430 [28.418, 28.441] | 37.337 [37.325, 37.349] | 0.762 [0.761, 0.762] | 30 | FAIL twice |
| c1 throughput | 26.736 [26.721, 26.752] | 32.744 [32.646, 32.830] | 0.817 [0.812, 0.821] | 6 | FAIL twice |
| c1 TTFT p50 | 319.709 [319.387, 320.019] | 178.614 [176.747, 181.373] | 0.555 [0.548, 0.562] | 30 | FAIL twice |
| c1 TTFT p90 | 319.709 [319.382, 320.016] | 178.614 [176.658, 181.230] | 0.555 [0.548, 0.562] | 30 | FAIL twice |
| c1 TTFT p99 | 319.709 [319.380, 320.006] | 178.614 [176.699, 181.170] | 0.555 [0.548, 0.562] | 30 | FAIL twice |
| c1 ITL p50 | 34.940 [34.922, 34.963] | 29.275 [29.258, 29.288] | 0.838 [0.835, 0.839] | 6 | FAIL twice |
| c1 ITL p90 | 35.368 [35.279, 35.449] | 29.610 [29.585, 29.635] | 0.838 [0.833, 0.841] | 6 | FAIL twice |
| c1 ITL p99 | 36.049 [35.863, 36.208] | 30.237 [30.106, 30.377] | 0.836 [0.829, 0.856] | 6 | FAIL twice |
| c4 throughput | 49.422 [49.400, 49.446] | 94.650 [94.573, 94.718] | 0.522 [0.521, 0.523] | 6 | FAIL twice |
| c4 TTFT p50 | 1180.663 [1142.596, 1218.225] | 512.854 [510.265, 515.502] | 0.413 [0.403, 0.463] | 30 | FAIL twice |
| c4 TTFT p90 | 1354.069 [1348.973, 1358.997] | 514.291 [508.378, 519.654] | 0.379 [0.370, 0.389] | 6 | FAIL twice |
| c4 TTFT p99 | 1354.069 [1348.847, 1359.042] | 514.291 [508.314, 519.818] | 0.379 [0.370, 0.389] | 6 | FAIL twice |
| c4 ITL p50 | 71.255 [71.251, 71.259] | 38.293 [38.284, 38.301] | 0.537 [0.537, 0.538] | 30 | FAIL twice |
| c4 ITL p90 | 71.440 [71.432, 71.447] | 38.555 [38.544, 38.567] | 0.540 [0.539, 0.540] | 30 | FAIL twice |
| c4 ITL p99 | 79.529 [78.874, 80.133] | 39.151 [38.982, 39.357] | 0.494 [0.482, 0.498] | 30 | FAIL twice |
| pp16384 | 305.433 [305.207, 305.687] | 428.487 [428.201, 428.743] | 0.712 [0.712, 0.715] | 6 | FAIL; confirmation pending |
| tg128@4096 | 27.656 [27.647, 27.664] | 36.434 [36.423, 36.445] | 0.759 [0.759, 0.760] | 30 | FAIL; confirmation pending |

### Pending coverage

The eight-hour GPU measurement budget stopped further collection. Every item
below also needs a new candidate identity after the host binary changed at
`8e4f0bc9`; the frozen `3bd3f609` records cannot complete that coverage.

| Workload | Absolute reference | MLX |
| --- | --- | --- |
| pp16,384; tg128@4,096 | Pending measurement | Primary FAILs above; confirmation pending |
| Serving c2/c8: throughput, TTFT and ITL p50/p90/p99 | Pending measurement | Pending measurement |
| Twelve-turn conversation: cold turn-1 TTFT, cached turn-2–12 TTFT, every turn’s decode and cache hits | Pending measurement | Pending measurement |
| Separate long group: tg128@32,768; pp65,536/131,072/262,016 | Pending measurement | Pending measurement |
| Peak RSS at 32K | Pending measurement support | Pending measurement support |

True peak RSS needs measurement-tooling support: native records report physical
footprint and HTTP records sample RSS at case boundaries; neither establishes an
RSS high-water mark. Long-context performance and conversation cache behavior
remain unqualified. There is no accepted full-profile baseline.

Frozen setup SHA256: `d432b45bdb5cb9e2eff6274776d655d0dfd5dad0b6a09f0e92eaefa04f505dff`.
Summary record SHA256: `dd06e4295eaa6c2974d83e2edeccc049a251627cda1446d9b64e6dbc68fca9f6`.
Raw artifacts, commands, records, validity audits and stop reasons are retained
with these identities. See [benchmarking](../../docs/benchmarking.md) for the
protocol, thresholds, noise checks and interval assumptions.
