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
