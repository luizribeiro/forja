# Add and ship an engine

Use [SmolLM2](../engines/smollm2/README.md) as the complete dense-transformer example.
Run commands from the repository root inside `nix develop`. Engines own model and
device choices; the host validates generic operations and memory capabilities.

1. **Confirm the architecture.** Read the upstream license, pinned `config.json`,
   tokenizer template and reference implementation. Save the revision and weight
   hashes before coding. Run `python3 engines/smollm2/download.py` for the example.
   Expect named dimensions matching
   [`Config`](../engines/smollm2/src/lib.rs), including tied embeddings and rotary
   conventions. Secondary model descriptions are clues, not a loading contract.
   Converted storage may differ from the reference state dict: verify affine
   dequantization, convolution axes, pre-adjusted norm weights and cast boundaries.
   Preserve source shards and hashes. When the checkpoint includes unused modalities,
   create a deterministic text-only grant by copying selected tensor bytes unchanged;
   test every rewritten offset and reject duplicate or invalid tensor spans.
   Select the pinned tokenizer JSON explicitly with `Tokenizer.from_file`; automatic
   loaders can rebuild a different tokenizer even when the template bytes match.

2. **Create the guest crate.** Copy the layout of
   [`Cargo.toml`](../engines/smollm2/Cargo.toml) and
   [`lib.rs`](../engines/smollm2/src/lib.rs). Use the SDK `Engine`, `Load` and
   `Weights` interfaces; keep config, tensor names and architecture in the guest.
   Run `forja build smollm2-135m-instruct.metal-apple-m3-ultra.reference` as a
   working example. Expect a component under `target/engines/`.
   Validate dimensions and checked offsets before allocation: invalid views must
   fail at construction, before they can reach a kernel.

3. **Make the plain model correct.** Follow
   [`reference.rs`](../engines/smollm2/src/reference.rs): embedding, decoder layer,
   attention, MLP, RMS norm, output. Use a contiguous cache and fixed portable
   algorithms. Return last-token logits and the documented hidden-state taps.
   Run `cargo nextest run --profile ci --release -p forja-host --test host -E 'test(smollm2::cpu_)'`
   for the random-weight cases in [`smollm2.rs`](../crates/forja-host/tests/smollm2.rs).
   Expect reference and optimized logits and every layer tap to agree under the
   shared normwise f32 tolerance, including cache continuations. This CPU check
   needs no model download. Run Metal cases through nextest's serialized group.

4. **Record independent goldens.** Run
   `uv run --locked --project support/golden support/golden/smollm2.py --model "$FORJA_MODELS/SmolLM2-135M-Instruct" --out "$FORJA_MODELS/golden/smollm2-135m-instruct"`.
   [`smollm2.py`](../support/golden/smollm2.py) uses transformers-f32 on CPU,
   records libraries and hashes, and produces four full-layer prompts plus 32
   greedy steps each. Run `forja verify --engine smollm2-135m-instruct.metal-apple-m3-ultra.reference --fixtures "$FORJA_MODELS/golden/smollm2-135m-instruct"`.
   Expect all layer, distribution and token checks to pass. A fixture generator
   must be independent of your guest so it can expose architecture mistakes.
   Pin a reference version that actually implements the architecture, in a separate
   requirements file when necessary; see
   [the hybrid reference requirements](../support/golden/qwen38-requirements.txt).
   Record that environment with the fixtures. Every prompt entry requires
   `prompt_tokens`, including fixtures generated from converted quantized storage.
   Match reference operation order, dtype and rounding points: algebraic rewrites
   can change greedy choices. Test negative inputs and threshold boundaries for
   scalar functions, and enforce reduced-precision agreement per prompt.
   Per-operation normwise limits are f32 1e-5, f16 2e-3, bf16/quantized 1e-2.
   Full f32 models require hidden error <= 2e-2, identical greedy tokens, and mean
   logit KL <= 5e-3. Reduced precision uses reference-token teacher forcing:
   mean KL <= 1e-2, top-1 >= 95%, hidden error <= 5e-2; free greedy is informative.

5. **Declare profiles.** Start with
   [the reference profile](../engines/smollm2/profiles/smollm2-135m-instruct.metal-apple-m3-ultra.reference.toml).
   Pin model revision/hash, storage precision, context and workload. A different
   precision is a different performance baseline. Path selection belongs in a
   profile, so both implementations remain compiled and testable in one artifact.
   Pair the instruction engine with
   [its chat component](../chat-formats/smollm2-chat/src/lib.rs); test exact template
   bytes, stop ids and malformed/unsupported requests before serving.

6. **Choose a measured optimization.** Use `forja profile` and `forja analyze`,
   then `forja bench-kernel` as described in [benchmarking](benchmarking.md).
   Implement variant picks and routing in
   [`optimized.rs`](../engines/smollm2/src/optimized.rs), never in the host.
   Expect a compatible variant at every dispatch site and parameter boundary.
   Kernel diagnostics identify a candidate; only `forja bench` measures its
   whole-model effect. Follow the measured optimization sequence in the
   [worked example](../engines/smollm2/README.md).

7. **Check replay and cache behavior.** Keep capture and bounded chunk planning in
   [`replay.rs`](../engines/smollm2/src/replay.rs). Compare optimized and reference
   logits/taps on identical weights, then verify the actual component against
   goldens. Run `cargo nextest run --release -p forja --bin forja -E 'test(metal_smollm2)' --run-ignored only`.
   Expect the [component verification tests](../crates/forja/src/verify.rs) to pass.
   Include page crossings, exact tails, reused graph shapes and context
   refusal. A numerically close result does not prove bitwise chunk invariance.
   If an engine declares `prefill_chunk_multiple`, add the catalog reduction-class
   test: every selected variant across merged/chunked shapes must share a proven
   reduction class. Bind that test to the real catalog as in
   [`catalog.rs`](../crates/forja/tests/catalog.rs), so a renamed or changed catalog
   entry cannot silently invalidate the claim. SmolLM2 declares `None` in
   [`lib.rs`](../engines/smollm2/src/lib.rs) and retains fixed serving boundaries.

8. **Benchmark and accept.** Run the [default fast gate](benchmarking.md) against
   frozen artifacts; use the exhaustive fixed-repetition gate for full audits.
   Save raw records and configuration, then accept only exact matching, complete
   profile measurements. Run `forja bench --engine smollm2-135m-instruct.metal-apple-m3-ultra.f32 --reps 30 --no-early-stop --json candidate.json`.
   Expect a provenance-bearing record with wall and GPU confidence intervals.
   Link results from the engine README to
   [`bench/`](../engines/smollm2/bench/). Never turn correctness-test timing into a
   performance claim: those tests can use small models or contexts.

9. **Ship a reproducible example.** Run the commit hooks and the guarded pre-push
   suite (`pre-commit run --all-files --hook-stage pre-push`), then follow the root [quickstart](../README.md) from a clean checkout.
   Expect a served completion with the pinned model and profile. Keep a short
   engine entry in [the index](../engines/README.md), serving settings in
   [serving](serving.md), and implementation-specific porting details beside the
   engine. Keep guides focused on reproducible commands and their expected results.
