# Benchmarking

Run commands inside `nix develop`. Keep frozen binaries, components, configurations,
model/token hashes, raw records and logs outside `target/`. Serialize GPU work.
Correctness and serving exactness remain required independently of timing selection.

## Classic and accelerated modes

Native `forja bench` uses fixed repetitions. The explicit classic spelling is:

```console
forja bench --engine /path/engine.wasm --model /path/model --reps 30 --no-early-stop --json record.json
```

The existing result payload, configuration hashes and rep counts are retained.
`measurement_mode` adds provenance; `--raw-samples` optionally adds every measured
rep's wall/GPU times and measurement window. Warmups are excluded from samples.

`support/bench_gate.py` provides accelerated gates. Every control is independent:

| Control | Accelerated behavior | Explicit classic behavior |
| --- | --- | --- |
| Sampling | Randomized sequential blocks, cap 30 reps/leg | `--no-early-stop --reps 30` |
| Rows | Compare dependencies; drift-check unaffected rows | `--all-rows` |
| Competitor | Reuse matching, fresh records with periodic probes | `--fresh-competitor` |
| Acceptance | `--accept --accept-from candidate.json` | `--accept --remeasure` |

```console
python3 support/bench_gate.py plan.json --out /path/evidence/accelerated
python3 support/bench_gate.py plan.json --out /path/evidence/classic --no-early-stop --reps 30 --all-rows --fresh-competitor
```

`--plan-only` writes the frozen plan, selection and modes without measuring.
Output directories must be new. A failed command stops execution. Fixed mode
preserves the four ABBA legs and their original confidence intervals; it does not
invent a pooled interval or a sequential PASS.

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
Confirmation records are fresh per row. Generic command adapters currently launch
per leg; loading, warmups and switching remain part of observed gate wall time.

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
Missing proofs, unknown dependencies or host/ABI/build changes select every row.
Selection exceptions include their type, message and traceback in the report.

Unaffected rows get five randomized blocks. Promote when their median log score
falls below `-D`, with D derived from raw block noise, never marginal noise.
Without comparable raw history, promote immediately: the first gate collects
history across all rows. Promoted rows start fresh. Do not reuse screening samples
for sequential decisions. `history.json` preserves dated block scores for future plans. History must have
a pinned artifact hash, matching identity/method, at least six finite scores and
nonzero SD. Stale/undated history (seven days by default) supplies no D; identity
or hash mismatches are refused. This conservative collection run does not accept
a candidate. A later independent run can use its verified noise history.

## Gate plan

A plan contains `repository`, `guard` (an argv prefix), frozen `artifacts` mapping
absolute paths to SHA-256, optional snapshot paths, and `rows`. Each row declares
`id`, `profile`, `question`, `X`, `Y`, `alternative` and baseline/candidate adapters:

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
A gate plan can supply `competitor_request` and `competitor_cache` paths.

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
