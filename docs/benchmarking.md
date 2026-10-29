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

## Choose the next optimization

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

Use `--mode decode --context 8192` for a decode step with populated KV. Prefill
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
perturbation remains unknown. Step shares use unprofiled wall time; host overhead
is therefore visible as unassigned time. Traffic counts logical operands, not
hardware DRAM transactions. Routed expert reuse produces a footprint interval;
proven unique routes tighten it. Padding/tile FLOPs are separate from logical
matrix FLOPs. Scalar/SFU/integer throughput has no calibrated ceiling.

Decode's model bound streams touched matrix weights plus KV once. Prefill uses
aggregate memory/compute rooflines. Both include the physical dispatch-count floor;
per-dispatch bounds additionally constrain serial execution. These optimistic
bounds omit scalar limitations and cache/occupancy effects. Efficiency above 100%
or a dispatch below its floor signals a probe/model/timestamp mismatch to
investigate, not negative recoverable time. Threadgroup bandwidth is recorded for
kernel investigation but is not used as a logical global-memory ceiling.

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
