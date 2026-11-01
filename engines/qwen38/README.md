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
| Hello | 1 | 31/32 | 0.0008287 | 0.01589 |
| What is 2 + 2? | 8 | 32/32 | 0.0001693 | 0.01518 |
| The capital of France is | 5 | 32/32 | 0.00008628 | 0.01190 |

Operator differentials use normwise relative error at most `1e-5` for F32 and
`1e-2` for BF16/quantized tensors; full-model
checks require mean KL at most `1e-2`, top-1 agreement at least 95% for each prompt,
and per-layer error at most `5e-2`. These are correctness results. No performance is claimed.

The diagnostic F32-KV profile retains q4 weights and BF16 activations. It passes
the same three per-prompt gates (31/32, 32/32, 32/32 choices); the arithmetic
fixture also passes with replay enabled.

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

## Porting notes

Quantized projections use one tiled reduction class. Inputs shorter than 16 rows
are zero-padded and cropped so every decode bucket uses the same row reduction.

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
