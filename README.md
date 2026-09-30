# Forja

Forja is a trusted Rust host for running LLM-written inference engines as WebAssembly components while executing their GPU work through Metal 4 with capability-controlled memory access.

This project is under construction.

Weight grants are capabilities for operator-owned model files. Granted files must not be modified
or truncated while a guest or tensor can still reference them. Breaking that contract can change
inference inputs or terminate the host with a bus error.

Measure Metal submission latency with:

```console
nix develop -c cargo run -p forja-metal --release --example submission_latency
```

Generate the Qwen3 transformer golden fixtures once after downloading the model:

```console
nix develop -c sh -c 'uv run --project support/golden support/golden/generate.py --model "$FORJA_MODELS/Qwen3-0.6B" --out "$FORJA_MODELS/golden/qwen3-0.6b"'
```

Run the pp512/tg128 Forja and MLX-LM engine baselines and print their median wall throughput with 95% confidence intervals:

```console
nix develop -c sh -c 'cargo run --release -p forja -- bench --engine "$FORJA_ENGINE" --model-dir "$FORJA_MODELS/Qwen3-0.6B" --json /tmp/forja-bench.json && uv run --locked --project support/bench-mlx support/bench-mlx/bench.py --model-dir "$FORJA_MODELS/Qwen3-0.6B" --json /tmp/mlx-bench.json && python3 support/bench-mlx/compare.py /tmp/forja-bench.json /tmp/mlx-bench.json'
```

Set `FORJA_ENGINE` to the Qwen WebAssembly component to measure. To compare replay behavior,
pass the default and `no-replay` components with two `--engine` flags.

Use `-c bench/suites/default.toml` for release-comparable measurements. For large models,
`-c bench/suites/big-model.toml` keeps the same workloads and context probes with fewer
repetitions for faster optimization-loop measurements.
