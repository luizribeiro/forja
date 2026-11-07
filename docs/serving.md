# Serving and configuration

Run Forja on macOS with a Metal 4 device. Enter `nix develop` before building or
running commands. `FORJA_MODELS` names the directory containing downloaded model
directories; the dev shell defaults it to `~/.cache/forja/models`.

1. **Build the engine and chat component.** Follow the root [quickstart](../README.md).
   `forja build smollm2-135m-instruct.metal-apple-m3-ultra.f32` writes the engine
   and its pinned text-only ChatML component under `target/engines/`.
   Model downloads and licenses are described in the [engine README](../engines/smollm2/README.md).

2. **Choose explicit settings.** Layer `-c path/to/config.toml` files in order,
   then `--set key=value` overrides. `--isolated` skips user configuration.
   For development commands, `forja config show --origin` prints values and origins.
   For serving, validate the layered file with `forja -c serving.toml config check --host`;
   `forja config schema --host --json` describes accepted keys. Profile context and
   precision are contracts, so an unsupported context is refused rather than clipped.

3. **Start serving.** Use the command in the quickstart or put its values in a file:

   ```toml
   [model]
   profile = "engines/smollm2/profiles/smollm2-135m-instruct.metal-apple-m3-ultra.f32.toml"
   engine = "target/engines/smollm2-135m-instruct.metal-apple-m3-ultra.f32.wasm"
   model = "/absolute/path/to/SmolLM2-135M-Instruct"
   chat_format = "target/engines/smollm2-135m-instruct.metal-apple-m3-ultra.f32.chat-format.wasm"
   ```

   Run `forja -c serving.toml serve`. Expect the configured HTTP listener and use
   `/v1/chat/completions` with the model id `HuggingFaceTB/SmolLM2-135M-Instruct`.
   The small model emits plain assistant text; it has no constrained tool protocol.

4. **Set the deployment boundary.** `forja serve` provides no authentication.
   Put authentication and tenant isolation in a trusted upstream service before
   exposing the listener to other users.

5. **Choose prefix reuse deliberately.** By default, paged-cache engines reuse
   exact token prefixes from earlier requests. `cached_tokens` and response
   timing reveal membership in a bounded LRU index of recently served prompts.
   Set `serving.prefix-reuse = false` for tenants sharing a server unless the
   trusted upstream provides the necessary isolation. Contiguous caches work,
   but do not reuse prefixes. This choice changes privacy and latency together.
   Hybrid engines with snapshot retention recompute replies while idle; incoming
   requests preempt at the next canonical block boundary.
   `forja_canonical_recompute_*` metrics expose pending work, completed jobs,
   blocks, preemptions, cancellations and failures.

6. **Keep weight grants stable.** Granted weights are capabilities for
   operator-owned files. Do not modify or truncate files while a guest or tensor
   may reference them: doing so can change inference inputs or terminate the host
   with a bus error. Replace a model by loading a new immutable directory and
   letting existing references drain.

The host bounds Metal compilation with `limits.pipeline_compilations` (256 attempts
per backend lifetime) and `limits.compile_budget` (`"60s"` compiler wall time
per engine load). Cache hits cost no attempts; failed compiles
consume their attempt. Budget expiry disables that backend's compilation and GPU
submission. These are operator settings; engine components cannot raise them.

Metal admits work using conservative device calibration and the actual encoded
launch schedule, in addition to the logical work and dispatch-count limits. With
the default ten-second deadline, forecasts are limited to two seconds per dispatch
and command buffer and eight seconds per submission. Shorter host deadlines lower
these ceilings and bound waits for resources. Engine components cannot raise them.
Each buffer also pays a calibrated startup reserve. Admission calibrates cold and
warm device observations, retains their raw rates, and reuses an exact source,
device and OS-build match for at most one hour under
`~/Library/Caches/forja/work-admission/`. Removing that directory forces calibration
in a fresh host process.

An oversized copy, RMS normalization, plain two-dimensional matrix or attention
query dispatch can split along output rows. Reductions retain their full K
dimension and order; attention retains its full cache and absolute query positions.
Other oversized operations are refused. The host preflights the whole submission,
then commits buffers sequentially, checking completion, device flags and the
measured/predicted ratio before committing the next buffer. A ratio above four,
a missing timestamp or timeout quarantines that encoded pipeline composition for
the current load. Valid completed outputs remain readable; three independent
failures of the same composition suspend the load. A timeout followed by a valid
completion retracts its uncertain evidence. A new load clears the quarantine.

Graph loading prepares the finite catalog specialization domains and generic
addressing fallbacks. Pipeline identity includes full skeleton source, math mode,
canonical constants and hook hashes. Prewarming reduces first-use latency. Cache
misses, including those during replay and lazy graph creation, can compile within
the pipeline quota and the current load's remaining compiler allowance. Starting
another load refreshes the wall-time allowance; it does not replenish pipeline
attempts or give eviction credit.

See [benchmarking](benchmarking.md) for serving workloads, prefix-cache evidence
and competitor comparisons, and [architecture and safety](architecture.md) for
the boundary between the host, engine and chat component.
