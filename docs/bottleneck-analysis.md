# Choosing the next optimization

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
