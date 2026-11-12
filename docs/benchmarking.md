# Benchmarking and optimization

Run commands inside `nix develop`. Preserve frozen binaries, components,
configurations, model/token hashes, raw records and logs outside `target/`.
Serialize GPU workloads. Correctness and serving exactness are required
independently of which timing rows are selected.

## Choose a measurement mode

Use the **default fast gate** for a reviewed candidate against frozen baseline
artifacts. It uses sequential stopping, automatic affected-row selection with
five-block drift checks on unaffected rows, a matching competitor cache, and
acceptance from the saved candidate record. Missing noise history collects all
30 blocks and reports `history-collected`; the cap is inconclusive, never PASS.

Use the **exhaustive fixed-repetition gate** for an audit, a fresh baseline, or a
suspected dependency/cache issue. `--reps N --no-early-stop` fixes repetitions,
`--all-rows` selects every row in the requested group, and `--fresh-competitor`
remeasures the competitor. `forja bench --accept --remeasure` also remeasures
instead of adopting a saved record. These controls are independent; use all of
them for the complete exhaustive workflow. Long-context work remains separate.

Native `forja bench` itself uses fixed repetitions; sequential decisions belong
to `support/bench_gate.py`. Every native record includes `measurement_mode`;
gate plans/reports preserve sampling, row-selection, process, cache and acceptance
modes and explicit fallback reasons. Cached competitor records retain their age
and original provenance. An adopted record retains `remeasured:false` and its
adoption hash. Inspect these fields before comparing results.

```console
forja bench --engine /path/engine.wasm --model /path/model --reps 30 --no-early-stop --raw-samples --json record.json
python3 support/bench_gate.py plan.json --out /path/evidence/fast
python3 support/bench_gate.py plan.json --out /path/evidence/exhaustive --reps 30 --no-early-stop --all-rows --fresh-competitor --process-per-leg
```

| Control | Default fast gate | Exhaustive controls |
| --- | --- | --- |
| Execution | Fresh native A/B pair retained for each block | `--process-per-leg` |
| Sampling | Randomized sequential blocks, cap 30 reps/leg | `--no-early-stop --reps 30` |
| Rows | Compare dependencies; drift-check unaffected rows | `--all-rows` |
| Competitor | Reuse matching, fresh records with periodic probes | `--fresh-competitor` |
| Acceptance | `--accept --accept-from candidate.json` | `--accept --remeasure` |


`--plan-only` writes the frozen plan, selection and modes without measuring.
Output directories must be new. A failed command stops execution. Fixed mode
preserves the four ABBA legs and their original confidence intervals; it does not
invent a pooled interval or a sequential PASS.

Native commands include batch sizes declared by their benchmark suite even when a
consumer reads only prefill/decode results. Use `--batch-measurements omit` explicitly
when no selected row consumes those batch cases; `include` is the fixed-repetition default.
Omitted coverage appears in mode/result metadata and changes performance hashes.
Such partial records cannot become accepted full-profile baselines. Use a complete
record or `--accept --accept-from partial.json --remeasure` to measure full coverage.
A resident worker with omitted batches retains only the single-request instance.

## Sequential decisions

Each block randomly uses ABBA or BAAB, with the seed and actual orientation saved.
Interpolate log scores at measurement-window midpoints in both directions and
symmetrize their difference. This cancels linear drift and reverses the sign under
label swaps even with unequal spacing. Scores are throughput or reciprocal latency.
The estimand is the exponentiated median block log ratio, not a ratio of marginal
medians. Even-sample geometric and arithmetic median conventions differ.

| Complete blocks | 6 | 10 | 16 | 20 | 25 | 30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lower order-statistic rank | 1 | 1 | 2 | 3 | 7 | 9 |

The upper rank is `n-k+1`. Exact fair-sign recursion gives first-crossing error
0.02499963529407978 per side, less than 5% total per row, including optional
stopping. This requires independent signs with a common median at the boundary;
arbitrary drift or serial dependence is not covered. No gate-wide simultaneous
95% claim is made. Gaussian power projections are diagnostics, not coverage proofs.

Each row declares `X` (default 0.02), `Y` (default 0) and a power alternative.
Non-regression requires a lower bound above `1-X`; improvement requires it above
`1+Y`. Latency thresholds must be converted explicitly. Equality never crosses a
boundary. Cap means inconclusive, never acceptance. Equivalence requires the
whole interval inside the declared band and is reported as `no change`.
Equivalence is an A/A diagnostic only: its two-sided decision error rates have
not been calibrated. It never contributes to a gating PASS, including in a mixed
diagnostic/gating plan.

A failed non-regression row gets exactly one fresh confirmation. Only FAIL+FAIL
confirms a regression; retain both records. Disagreement remains unconfirmed.
Improvement requires the original PASS, with no rescue run. Drift alarms can void
PASS, never erase FAIL. The alarm uses verified history-derived D, not X.
Invalidated PASS is not automatically retried. Without verified D, a would-be
gating PASS continues through all 30 blocks and reports `history-collected`, never
PASS. A/A diagnostics explicitly label unavailable drift protection.

Rows sharing identical frozen commands reuse whole sessions: a conversation's
turns and a concurrent case's requests are observations of that session, not
independent trials. Extend the session sequence while another row needs samples.
Confirmation observations are fresh per row.

Each sequential block starts a fresh process per side with its frozen workload.
Startup order is randomized independently of ABBA/BAAB and recorded with its seed.
Both workers finish loading before the first leg and stay loaded for all four
legs. Process-level variability belongs inside the independent block. A fixed
pair across blocks can confound process effects with build effects. Rows with
matching commands share already-collected blocks; only one pair is retained. Each leg runs
the declared warmups, drains GPU work and releases paged request lanes between
measurement phases. Prefix workers share one model instance across serial single
and batch phases. They prepare authenticated read-only prefixes once per worker
and context, fork them per leg, and refuse prefix rebuilds within the block.
Untimed decode priming is bounded to two steps per shape; prefill primes once at
its declared shape. Full measured workloads stay unchanged. Classic process calls
retain their original warmup policy. Loaded weights and compiled kernels remain resident. Native records identify the worker PID, load count and
cache policy; group records retain startup commands/times and complete worker logs.
Per-leg command records distinguish request wall time from worker startup.
Each block records total wall time including both starts and cleanup, and each
worker records its PID and start/end timestamps.

Use `--process-per-leg` for the launch-based path, older binaries without the worker
protocol, or process-cold measurements. A native adapter may declare
`"cache_state":"cold-process"` to require this path for its whole group;
`"fresh-request"` is the default. Unknown policies are errors. Fixed ABBA mode also
uses the original per-leg native calls. Non-native command adapters retain their
own process/session policy, recorded with the fallback reason.

`forja bench --worker --raw-samples` serves newline-delimited JSON requests on
stdin, each `{"id":1,"output":"/absolute/new-record.json"}`. IDs increase from one.
The CLI freezes one engine, point, selection, repetition count and warmup policy;
requests cannot change configuration. `FORJA_BENCH_READY` and `FORJA_BENCH_RESULT`
prefix structured control lines on stdout; EOF closes the worker. The gate client
checks PIDs, one model load and fixed prefix build counts throughout. Worker startup permits the documented
45-minute macOS launch queue; `--worker-timeout` bounds each sample response.
Loading, warmups, resets, IPC and switching all remain part of observed gate wall
time. Report startup and steady request times separately, plus amortized totals.

## Build and row selection

```console
forja bench-snapshot --engine /path/engine.wasm --json build.json
python3 support/gate_snapshot.py build.json --proof coverage.json --json snapshot.json
python3 support/gate_select.py baseline.json candidate.json scoreboard.json --json selection.json
```

Use one profile per snapshot pair. The binary's production resolver supplies picks,
including overrides. Source fingerprints bind its revision, compiler flags,
toolchain, profile and catalog. Metal's concatenated shader module is fingerprinted
as a whole, conservatively affecting every consumer when any constituent changes.

A guest coverage proof binds `guest_source_hash`, lists all resolved `sites`, and
maps each scoreboard row to `{"site":...,"ranges":{"position":[lo,hi],...}}`
paths. Audit and test the proof against guest routing helpers. Inclusive ranges
must cover every query/chunk boundary, generated position, graph and batch bucket;
end-to-end rows include prefill/cache/decode consumers. Traces cannot prove absence.
Missing proofs, unknown dependencies or host/ABI/build changes conservatively
select regular rows. These fallbacks do not establish a source-bound affected long
row. Selection exceptions include their type, message and traceback in the report.

Unaffected rows get five randomized blocks. Promote when their median log score
falls below `-D`, with D derived from raw block noise, never marginal noise.
Without comparable raw history, promote immediately: the first gate collects
history across all rows. Promoted rows get a fresh primary measurement namespace, including when other
rows have already collected shared primary observations. Do not reuse screening samples
for sequential decisions. `history.json` preserves dated block scores for future plans.
Noise identity contains exactly `row`, `workload_sha256`, `harness_protocol`,
`machine_identity`, `os_build` and the plan’s `guard` argv prefix. The workload hash binds model and token inputs,
workload settings and metric/window selection. The versioned harness protocol also
binds effective execution, priming and warmups. Changing a workload, protocol,
machine, OS or resource wrapper invalidates reuse; changing host/engine code does not. Binary,
component, profile, source and command-path hashes belong to measurement provenance,
not noise identity. Frozen measurement artifacts still require exact hashes.

History needs a pinned SHA256, matching identity/method, at least six finite block
scores and nonzero SD. Stale/undated history (seven days by default) supplies no D;
identity or hash mismatches are refused. Older build-bound histories additionally
require `history.legacy_workload_sha256`: an audited hash of their original workload
and tokens, compared with the current workload, plus the audited original `history.legacy_guard` argv. This explicit conversion preserves
the pinned original file, timestamps and scores; never guess the missing hash from
the new build or retag old measurements as new ones.

`history.sd_factor` declares the allowed SD ratio (default 2, finite and greater
than one). At the first sequential look, and after each later block, compare the
observed block SD with stored SD in both directions. The five-block unaffected-row
screen also checks it. A departure outside `[SD/factor, SD*factor]`, a fired drift
screen, or baseline drift irreversibly forces full 30-block collection for that
row, including any required confirmation. Reports and saved histories retain the
reason, phase, block count, factor and observed/stored SD. A guarded PASS cannot
be accepted or rescued later; an earlier FAIL remains evidence and still gets
exactly one fresh confirmation. Missing noise history likewise collects all blocks
before any positive decision, with no protected PASS. A later independent run can
use the resulting verified workload noise.

## Separate long-context gate

Every row declares a positive integer `workload_max_context`: an audited ceiling
on the context actually prepared or consumed by its **whole command**, including
priming, other metrics, suites and generated tokens. A configured cache capacity
alone is not a prepared workload. Unknown ceilings are refused. Explicit native
prefill/decode axes also raise the ceiling; rows sharing a command share its
largest ceiling. A short metric collected alongside 64K decode therefore belongs
to the long group too.

A ceiling of 32,768 tokens or more routes the row to `long`, including 32K/64K/128K
decode and 64K/128K prefill. Everything below it belongs to `regular`.
The default invocation runs only `regular`. `--all-rows` applies within the
requested group and never folds long work into a regular decision.

```console
python3 support/bench_gate.py plan.json --group regular --out /path/regular
python3 support/bench_gate.py plan.json --group long --out /path/long
python3 support/bench_gate.py plan.json --group long --long-reason "periodic coverage" --out /path/periodic-long
```

A validated source-bound dependency proof marking a long row affected makes the
separate long group required (`required_by_proof`). Running `--group long` then
selects that group. Unknown dependency fallbacks and explicit all-row selection
cannot substitute for this proof. Otherwise a long run needs an explicit manual
or periodic `--long-reason`. Regular and long decisions always remain separate.
Saved plans and reports contain both `gate_groups`, each selected or
`skipped-with-reason`, and retain every excluded row. Reopening a saved regular
plan preserves the long group. A descriptive long screen must be labeled a
screen; it supplies neither a gate decision nor reusable sequential history.

## Gate plan

A plan contains `repository`, `guard` (an argv prefix), frozen `artifacts` mapping
absolute paths to SHA-256, optional snapshot paths, and `rows`. Each row declares
`id`, `profile`, `workload_max_context`, `question`, `X`, `Y`, `alternative` and baseline/candidate adapters:

```json
{
  "command": ["/path/frozen/forja", "bench", "--isolated", "--engine", "/path/engine.wasm", "--model", "/path/model", "--pp", "512", "--tg", "128", "--reps", "{reps}", "--set", "bench.warmups={warmups}", "--raw-samples", "--json", "{output}"],
  "metric": "/results/0/pp/tokens_per_second/wall/median",
  "window": "/results/0/raw_samples/pp/0/measurement_window",
  "expected_provenance": {"/provenance/binary_sha256": "...", "/inputs/0/engine_sha256": "..."}
}
```

Placeholders replace whole arguments; `bench.warmups={warmups}` is also supported.
Use absolute paths and record the effective quotas/configuration. `reciprocal:true`
converts a latency metric to a score. A source-bound history file needs its path,
SHA-256 and matching `history_identity`; otherwise block noise/power is unknown.
`require_equal_outputs:true` adds equality across the two implementations; native
determinism is checked within each implementation regardless. Serving adapters use
`--reps`, `--warmups`, `--raw-samples` and their case/turn measurement windows.

## Competitor cache

```console
python3 support/bench_cache_run.py request.json --cache /path/cache --json competitor.json
python3 support/bench_cache_run.py request.json --cache /path/cache --json fresh.json --fresh-competitor
```

The request declares `version_command`, `effective_config`, `model_files`,
`workload_files`, `tokens_files`, `measure`, `drift` and `metrics` (JSON pointers).
Commands are argv arrays with exactly one `{output}` argument. Include all model,
quantization, tokenizer, workload/driver and actual token inputs. The version
command runs on every lookup; content, machine and OS changes invalidate reuse.
Read the effective competitor configuration and verify its intended path before
running. The short drift command must use the same path/workload at fewer reps.

`--max-age` defaults to seven days, `--drift-interval` to one day and
`--drift-tolerance` to 0.05. A failed/malformed probe stops; detected drift invalidates
the cached measurement and triggers a full run. Reports retain original provenance,
measurement age, cache key and probe records; they never relabel cached work fresh.
A gate plan can supply `competitor_request` and `competitor_cache` paths, with an
audited `competitor_workload_max_context` for that whole request. A cache command
belonging to the other gate group is retained as `skipped_competitor` with its
reason; it cannot introduce long work into a regular run.

## Acceptance and kernel diagnostics

Run the exact frozen binary when adopting its saved candidate record:

```console
/path/frozen/forja bench --engine /path/engine.wasm --model /path/model --accept --accept-from candidate.json
/path/frozen/forja bench --engine /path/engine.wasm --model /path/model --accept --accept-from candidate.json --remeasure
```

Adoption refuses binary/component/model/profile/configuration hash mismatches,
dirty provenance and incomplete/duplicate results. It preserves the measured
workload and rep counts; choose the record intended for the accepted baseline.
The adoption hash and `remeasured:false` are retained. Remeasurement uses the
saved configuration with normal measurement and clean-source checks.

`forja bench-kernel VARIANT --shape shape.json --json diagnostic.json` executes
exactly that catalog variant (five reps, one warmup by default). The JSON specifies
an `operation` with `kind` and core fields, plus `inputs`/`outputs` with `dtype`,
`shape`, optional `permutation` and exact `u32_values` for metadata. Floating tensors
start at zero. Paged attention accepts explicit noncontiguous page tables and
generation arrays. Core operation/catalog constraints and allocation quotas apply.
Records say `kernel-diagnostic` and `model_benchmark:false`; use them for screens,
never as model benchmarks or accepted model baselines.
For recurrent diagnostics, `--state-steps N` chains scalar delta-rule or
single-token convolution updates through distinct destination buffers. Each
operand set holds its own chain; `--serial` also serializes the sets through the
shared first output. These are diagnostic tensor dependencies, using fixed test
authentication metadata, rather than engine state transactions. Records include
the chain length, total dispatch count and all resident destination bytes.
`--advance-position` visits consecutive token positions on each paged attention
operand set before moving to the next set, incrementing key lengths too. Its
dispatch count must be a multiple of the operand-set count. With append-and-attend
variants, this measures successive appends to the same pages. Allocate space for
every row. `operand_order` records `consecutive` for this flag and `cyclic`
otherwise. This flag nulls `cost_per_dispatch` and `rates`.
`--row-chunk N` splits ordinary dense or quantized matrix rows into consecutive
dispatches sharing each operand set's weights. The selected variant is validated
for every chunk. Row chunks cannot be combined with `--serial`, epilogues, append
copies, state chains, or advancing positions.
Records include the chunk limit and actual dispatch count; unequal tails omit rates.

An optional `append` array in the shape supplies two ordinary
`[1, kv_heads, head_dim]` tensors. Each visit records two authenticated copies plus
one attention dispatch. `primary_dispatches_per_sample` counts dispatches of the
requested variant, including all state steps; `dispatches_per_sample` also counts
the append copies. These composite diagnostics omit single-kernel cost/rate
estimates.

Per-dispatch rates require identical logical costs across all recorded operations.

Rates count logical operands and FLOPs rather than hardware transactions or
instruction throughput. Quantized matrix diagnostics also report compiled
execution width, threadgroup limits and static threadgroup storage; these limits
do not measure occupancy, register allocation or spills.

Metal kernel and model benchmarks check each completed command buffer before
writing a record. They require an existing calibration cache for the machine,
GPU and OS build. Run `forja calibrate` before starting any benchmarks or workers;
bench processes never create or refresh this calibration. Missing, corrupt or
mismatched caches fail with the cache path and a `forja calibrate --force` remedy.
Rates above the fastest raw calibration observation plus a **10% margin**
are refused with an `impossible benchmark record` error.

The arithmetic check counts unconditional dense matrix FLOPs against the fastest
supported calibrated arithmetic. The bandwidth check uses the largest contiguous
matrix, copy or argmax input **only when it exceeds the 512 MiB cache allowance**,
subtracting that allowance from the guarded byte count. Inputs at or below
512 MiB receive no bandwidth check. It never sums logical operand traffic across
reused tensors. Small cached inputs, noncontiguous views and routed or masked work
do not establish a physical bandwidth or arithmetic lower bound for this check.
Unknown or overflowing dispatch costs are omitted from the lower bound. Graph
summaries count only static dispatches and are cached across replay; dynamically
instantiated contributions are omitted. Missing GPU timestamps skip buffer
validation and increment `roofline_guard.buffers_without_gpu_timestamps` (a
process-wide counter including warmups). The diagnostic's logical rates remain
distinct from the guarded rates. Refusals suggest `forja calibrate --force` to
refresh the calibration.

`roofline_guard` records the limits, margin, cache allowance, calibration content
hash, machine identity and calibration provenance. Raw observations remain in the
calibration cache; `forja calibrate --json calibration.json` exports them.

## Choose the next optimization

Calibration records include dependent scalar arithmetic, special-function and
scattered-read probes for work admission.
These probes supplement the roofline ceilings; the roofline ranking still treats
scalar and integer instruction throughput as uncalibrated.

Use `forja calibrate` once per device/OS identity. Its cache includes GPU family,
core count, memory and OS build. `--force` refreshes it; `--json calibration.json`
saves raw observations, median confidence intervals and build/load provenance.
Streaming traffic, supported MMA arithmetic, threadgroup reads and dependent
copies are microbenchmarks, not model benchmarks. MMA configurations are screened
with independent accumulator counts, then confirmed with separate observations.
BF16 compiler refusal is labelled as f32 promotion rather than native BF16.

Record the actual engine component and full model with a costed profile:

```console
forja profile --engine /path/engine.wasm --model /path/model --mode prefill --context 16384 --json --set 'paths.scratch="/path/evidence/pp16k"'
forja analyze --profile /path/evidence/pp16k/profile.json --calibration calibration.json --json analysis.json
```

Profile submissions retain raw admission calibration and each command buffer's
forecast, startup reserve, GPU counter interval and backstop threshold. Analysis
reports measured/predicted ratios from these unscaled intervals; a submission
average cannot hide an individual buffer overrun. Dispatch ratios exclude the
buffer startup reserve and serve as diagnostics rather than backstop decisions.
Legacy profiles without admission records retain their roofline analysis.

Use `--mode prefill --step-tokens 512 --context 16384` to profile a cold paged
serving lane with an engine-declared prompt bucket. Every step is retained. Add
`--prefix-context N` to measure only the suffix; cache preparation is reported
separately from the timed GPU phase.

Kernel diagnostic records also retain a separate precise work profile with raw
per-dispatch and buffer forecasts, intervals and calibration. This extra submission
is excluded from the diagnostic's regular samples.

Use `--mode decode --selection forward` or `--selection masked` to measure the
corresponding entry separately; setup and completion run outside that interval.
Use `--mode verify --step-tokens 3` to measure a declared target verification
shape, including target sampling. The engine must declare that shape under the
selected dispatch quota. These profiles are diagnostics, not benchmark records.

Use `--mode decode --context 8192` for a decode step with populated KV.
Add `--batch-size 4` to profile a declared paged decode bucket with private state
and populated KV per lane; the report counts all measured lane tokens. Context
priming follows the engine-declared numerical prompt blocks. Prefill
records retain every chunk, so an early chunk cannot stand in for the full run.
Omit `--calibration` to reuse or create the matching device cache. Identity/schema
mismatches are errors; profiles made by older binaries need recording again.

Start with the ranked call sites, grouped across layers/chunks. The report names
the selected variant, constraints-compatible alternatives, resource bound and
recoverable GPU time at the ceiling and at `--target 0.8` (the default). JSON also
contains individual dispatches, composite kernels, per-kernel totals, phase bounds
and layer-type efficiency. A composite's internal kernels replace its outer time
in kernel totals, so they are counted once. Candidate intersections satisfy every
instance in a group. Copy a candidate and representative validated shape into
`forja bench-kernel VARIANT --shape shape.json --json screen.json`; confirm any
promising change with a full model `forja bench` record before adopting it.

Interpret these as diagnostics. Precise timestamps perturb execution; the report
shows that ratio and scales attribution to an unprofiled GPU control. Per-kernel
perturbation remains unknown. Command-start sampling may overlap subsequent work,
so attribution uses the larger command-counter or precise span per submission.
Both time bases are retained; GPU attribution stays within the unprofiled control. Step shares use unprofiled wall time; host overhead
is therefore visible as unassigned time. Traffic counts logical operands, not
hardware DRAM transactions. Routed expert reuse produces a footprint interval;
proven unique routes tighten it. Padding/tile FLOPs are separate from logical
matrix FLOPs. Scalar/SFU/integer throughput has no calibrated ceiling. Floating-point counts
use an optimistic MMA-equivalent ceiling, including scalar FMA kernels; this is
not an instruction-throughput estimate. Sampling work describes the logical
stochastic path, so greedy or conditional branches may execute less work.

Decode's model bound streams touched matrix weights and KV once, then includes
the declared state, block scratch and authentication traffic. Prefill uses
aggregate memory/compute rooflines. Both include the physical dispatch-count floor;
per-dispatch bounds additionally constrain serial execution. These optimistic
bounds omit scalar limitations and cache/occupancy effects. Efficiency above 100%
or a dispatch below its floor signals a probe/model/timestamp mismatch to
investigate, not negative recoverable time. Threadgroup bandwidth is recorded for
kernel investigation but is not used as a logical global-memory ceiling.

Work totals retain scalar/SFU counts and state, scratch and authentication-byte
subsets. Subsets are already included in reads/writes; SFU operations are already
included in scalar work. Engine-declared layer types have GPU and wall shares.
Kernel groups use internal stages when measured, while call-site groups retain
the enclosing composite; do not add the two rankings together.

Register-risk annotations count source-level persistent carry/result values per
thread. They exclude temporaries and compiler allocation, so they identify a
possible spill risk rather than an observed spill or an occupancy prediction.
An absent annotation does not establish that a kernel cannot spill.

Private state snapshots currently use host read/write transfers. Their time is
outside GPU kernel attribution; cached-prefix preparation is reported separately.
GPU COW and ordinary copy commands count the transferred views, including partial
page tails. A suffix profile reports both full context and measured tokens;
decode reports one measured token. Use a profile with sufficient declared context
capacity to include the requested decode position.

A `[! >100%]` marker identifies a displayed percentage above 100%.
`[! outside calibration CI]` identifies an interval shorter than the optimistic
calibrated bound. A median-only bound warning stays visible within the calibration
interval. Model and layer-type flags use the GPU time displayed in their row;
resource-rate flags use raw profiled GPU intervals. Group markers retain flagged
individual intervals. Bound envelopes combine per-probe CI endpoints and do not
form a joint 95% interval.

Split routed projections each read their activation inputs. Sub-kernel traffic
intentionally counts those reads per projection; model bounds use the enclosing
logical cost. Shape-only minimum footprints assume distinct routes per row.
Profiles prove that guarantee from top-k producers and weaken unproved bounds.

## Reproduce the SmolLM2 records

Download and build as in the [quickstart](../README.md), then run a complete suite:

```sh
forja -c bench/suites/default.toml --set 'bench.selection=["gpu-pipelined"]' bench --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --raw-samples --json /absolute/path/forja.json
uv run --locked --project support/bench-mlx support/bench-mlx/bench.py --model-dir "$FORJA_MODELS/SmolLM2-135M-Instruct" --suite bench/suites/default.toml --json /absolute/path/mlx.json
python3 support/bench-mlx/compare.py /absolute/path/forja.json /absolute/path/mlx.json
```

Use the same suite and token generator, and compare the matching pipelined row.
The MLX driver records effective settings and validates the loaded precision.
SmolLM2's Forja profiles use f32-promoted weights, whereas this MLX baseline uses
native bf16: label that precision difference instead of calling it a matched
precision comparison. The [worked example](../engines/smollm2/README.md) links the
actual records and explains each measured optimization.

## HTTP conversations

`forja bench http` packages the same streaming client for either server. Supply a
suite, server argv JSON array and frozen identity JSON; it starts and stops only
that owned server. Use a Python environment containing the pinned Transformers
and tokenizer packages via `FORJA_BENCH_PYTHON`.

```console
forja bench http --suite conversation.toml --url http://127.0.0.1:8080 --server server.json --identity identity.json --reps 1 --warmups 1 --raw-samples --json conversation.json
```

The identity binds model/revision, tokenizer directory, named artifact paths and
SHA256 hashes, effective server settings and template options. Required artifacts
are `tokenizer`, `tokenizer_config`, `weights` and `server`. Additional shards,
components and configuration files should also be included. Optional
`cache_metrics` names the metrics URL, pending gauge and completed counter.

Each repetition starts a fresh server, performs `--warmups` two-token
initialization requests (default one), then runs the whole scripted conversation. Actual replies
are appended to subsequent histories. The original tokenizer renders and checks
every prompt against the suite's whole-command context ceiling before HTTP;
server prompt-token counts must agree. Per-turn records retain request history,
usage, prompt hashes, visible SSE events, TTFT, decode rate, cache hits and
recompute observations. Decode rate is `(completion_tokens - 1)` divided by the
interval from first visible content through DONE; SSE chunks are not tokens.
Recompute idle waits remain outside TTFT/decode and inside total session time.
A generated-reply hit means the reused prefix extends beyond the previous prompt;
a recompute hit additionally requires the configured recompute metrics. Unsupported
counters are null. Raw turns are flushed as they complete, including before a
later failure. Turns are observations within one independent conversation.

Use `--mode serving` with the concurrent serving-suite schema to drive the same
client's concurrency, cancellation and exactness checks. Its frozen identity also
supplies `repository`, `weights` paths and, for Forja, `resolved_config` plus the
binary/component/profile/chat-format artifacts. Managed server output is retained
beside each record. Greedy HTTP requests omit seed so server batching can run.

Stateful native engines that cannot restore authenticated prefix state use worker
protocol 3: weights stay resident for one block while every request creates fresh
state. Records distinguish this from protocol 2 prefix forks and include priming,
setup, measured and output-probe costs. Stateful prefill submissions respect the
profile's `prefill-chunk` bound, wait for scratch reuse between chunks, and time the
whole prompt. Decode priming follows declared numerical prompt blocks; records
made before a change to this policy must be measured again.

The MLX native adapter in `support/bench-mlx/native.py` implements this fresh-request
worker contract with one loaded model and newly constructed caches per request.
Its generator computes one token ahead: decode timing drains that lookahead after
one yielded token, then measures the declared input positions. Whole-command
context ceilings also include the extra prefill lookahead token. Classic process
calls retain every configured full warmup; worker priming stays bounded.
The optional HTTP server adapter applies a byte cap to the existing LRU cache
constructor as well as the server admission limit. Its separate `--audit-log` mode
records actual BatchGenerator request IDs and cache types. Collect admission
proof with observation enabled, then disable observation for timing runs and
retain the matching source/configuration identities.

### Scheduled completion records

`forja bench completion` runs rendered prompt files plus their declared suffixes through the serving
scheduler. Plain decoding is the default. `--mode draft` enables only the
embedded profile's prompt-lookup policy; it requires `--plain-record` from the
same model, component, tokenizer, prompt, seed and effective configuration.
A mismatched token stream fails the command and retains a record marked
`stream_check.status = "failed"`. The reusable check is:

```sh
forja bench check-streams --plain plain.json --draft draft.json
```

`bench/suites/prompt-lookup.json` defines the original prompt fixtures, their
sources, fixed seeds and thresholds. Padding uses 256 distinct synthetic background
documents from unrelated survey/catalogue topics; its deterministic generator
and seed are checked in. Every ordinary row records its own acceptance counts. Select regular rows with `--row NAME`;
select the separate long group with `--group long`. Long workloads reserve
32,512 prompt tokens plus 256 generated tokens within the profile's 32,768-token
limit. Records retain the actual rendered prompt length and hash.

```sh
forja bench completion --engine /path/engine.wasm --model /path/model --chat-format /path/chat.wasm --suite bench/suites/prompt-lookup.json --row code-edit-short --set limits.dispatches_per_list=8192 --reps 1 --json plain.json
forja bench completion --engine /path/engine.wasm --model /path/model --chat-format /path/chat.wasm --suite bench/suites/prompt-lookup.json --row code-edit-short --set limits.dispatches_per_list=8192 --reps 1 --mode draft --plain-record plain.json --json draft.json
```

Both modes use identical quotas and profile settings. `--cached` prepares a
prefix before each measured request; `cached_tokens` records whether reuse
actually occurred. It changes no lookup setting. A cached draft record requires
a correspondingly cached plain record.

Per-row observations include rounds, exact accepted-depth counts, mean accepted
drafts, k (zero in plain mode), committed effective-k counts, episode
entry/exit reasons, policy identity, GPU times, and ordered per-request token
hashes. GPU time labelled plain includes prompt work and excludes Verify.
Decode wall spans start after prompt completion; whole-group wall time includes
prompt execution. Repetitions are whole request groups, with raw windows and
median/95% confidence intervals. Token hashes use little-endian u32 IDs;
`output_digest` binds the ordered request hashes without logits. Records retain
finish reasons, actual output counts and each row's declared `min_output_tokens`.
A length finish requires the requested maximum; natural stop/EOS requires at least
that minimum and at most the maximum. The current text fixtures require 128 tokens
from file edits, copied/generated JSONL and explanatory replies; one-token prompt
rows require one. Refusals report every request's actual count and finish reason.
Prompt-token counts differing from service usage are refused. Decode throughput
uses actual generated tokens divided by decode wall; EOS is never ignored. Decode spans must
be positive. c4 and prompt rows gate **only on group wall time**; their decode
spans are diagnostic. Single-lane generation rows gate on decode wall time.

Reported k is the largest observed effective depth, and drafter identity comes
from admitted episode sources. A draft row without verify rounds records
`drafting: "never-entered"`, k zero and no observed drafter. The reusable stream
check refuses that state on any row with `minimum_speedup > 1`.

Performance hashes bind binary SHA-256, clean-source state and device. Stream
checking refuses dirty or incomplete provenance. Candidate checks require the
same binary and source. To compare a frozen baseline with a candidate explicitly:

```sh
forja bench check-streams --comparison baseline-vs-candidate --plain baseline.json --draft candidate.json
```

That mode permits source, binary, component, profile hashes and lookup policies to
differ; model, tokenizer, chat format, sampling, effective settings, device and OS
must match. Both modes require identical output token streams.
`bench completion --stream-comparison baseline-vs-candidate` labels its automatic
assertion likewise. Prepare a compatible clean plain reference before starting
workers, including when startup or leg order puts the draft worker first.

`--worker --raw-samples` supports immutable ordered sample requests using the
native worker protocol (version 1), retaining one loaded completion runner.
Each leg prepares fresh requests; prefix reuse is disabled in this mode.
Each leg warms the row's concurrency and full output budget (at least
`min-match + max-drafts + 2` tokens), identically in both modes. Records include
that shape, request count and observed warm-up rounds. Uncached long legs need
a full-context warm-up and another full prefill for the sample: retaining the
warm-up prefix would change the cache path being compared. Cached process runs
keep their declared reuse policy; worker prefix reuse remains disabled.
`--reps N --no-early-stop` retains fixed repetitions. Worker and process records
name their execution and decoding modes separately.

### Workload-based runtime planning

These are **planning estimates, not benchmarks**. The illustrative cost model
assumes 2,000 prompt tokens/s and 25 output tokens/s in aggregate, charging lanes
serially. Short ASCII prompt byte counts conservatively bound token counts;
other rows use their declared prompt budgets. For prompt allowance P, output O,
concurrency C and warm-up W=max(O,12), a leg costs
`C * (2*P/2000 + (W+O)/25)` seconds. This includes the second long-context prefill.
The table assumes one warm-up and one measured group per leg, four legs per
block, and all 30 blocks. It excludes model loading, preflight references, drift
probes and any regression confirmation. Actual elapsed time must be recorded.

| Row | Prompt allowance | Lanes | Estimated hours for 30 blocks |
| --- | ---: | ---: | ---: |
| code-edit-short | 992 | 1 | 0.72 |
| jsonl-copy-short | 1036 | 1 | 0.72 |
| code-short | 447 | 1 | 0.70 |
| chat-short | 437 | 1 | 0.70 |
| thinking-short | 788 | 1 | 0.71 |
| jsonl-generate-short | 433 | 1 | 0.70 |
| code-edit-4k | 4096 | 1 | 0.82 |
| jsonl-copy-4k | 4096 | 1 | 0.82 |
| code-4k | 4096 | 1 | 0.82 |
| chat-4k | 4096 | 1 | 0.82 |
| thinking-4k | 4096 | 1 | 0.82 |
| jsonl-generate-4k | 4096 | 1 | 0.82 |
| jsonl-copy-seeded | 4096 | 1 | 0.82 |
| code-edit-32k | 32512 | 1 | 1.77 |
| jsonl-copy-32k | 32512 | 1 | 1.77 |
| chat-32k | 32512 | 1 | 1.77 |
| thinking-32k | 32512 | 1 | 1.77 |
| c4-short | 437 | 4 | 2.79 |
| prompt-short | 437 | 1 | 0.03 |
| c4-4k | 4096 | 4 | 3.28 |
| prompt-4k | 4096 | 1 | 0.15 |

The size-only planning estimate totals 16.2 hours for regular rows and 7.1
hours for the separately selected long group (23.3 hours combined), before
startup and diagnostics. One fresh confirmation of every regression row could
double the affected portion; it is not part of the original acceptance run.
