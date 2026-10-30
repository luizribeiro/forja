# Qwen3.8-27B

Text-only inference for `mlx-community/Qwen3.8-27B-4bit`: 64 hybrid decoder layers,
affine q4/group-64 weights, BF16 activations, F32 recurrent state and F16 KV.
The Metal profile targets Apple M3 Ultra and limits context to 32,768 tokens.
Prompt processing uses canonical 64-token WY blocks and a final partial block;
decode advances recurrent state one token at a time. Single-lane SDK calls declare
`BatchPhase::Prompt` or `BatchPhase::Decode` explicitly. Prompt submissions start
on a 64-token boundary. Each new request clears its recurrent carry; graph replay
and cross-request cache reuse are disabled.

## Use

Run from the repository root inside `nix develop`:

```sh
python3 engines/qwen38/download.py
forja build qwen38-27b-4bit.metal-apple-m3-ultra.q4
forja run --engine qwen38-27b-4bit.metal-apple-m3-ultra.q4 --prompt 'The capital of France is' --max-tokens 32 --temperature 0
forja --isolated -c engines/qwen38/serve.toml --set "model.model=\"$FORJA_MODELS/Qwen3.8-27B-4bit\"" serve
```

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
The full WebAssembly component passes the reduced-precision end-to-end criteria:

| Prompt | Prompt tokens | Matching choices | Mean KL | Maximum layer relative error |
| --- | ---: | ---: | ---: | ---: |
| Hello | 1 | 31/32 | 0.0006813 | 0.01493 |
| What is 2 + 2? | 8 | 32/32 | 0.0001912 | 0.01415 |
| The capital of France is | 5 | 32/32 | 0.0001259 | 0.01498 |

Operator differentials use normwise relative error at most `1e-5` for F32 and
`1e-2` for BF16/quantized tensors; full-model
checks require mean KL at most `1e-2`, top-1 agreement at least 95%, and per-layer
error at most `5e-2`. These are correctness results. No performance is claimed.

Regenerate the fixtures with the pinned CPU reference environment:

```sh
uv run --no-project --python 3.12 --with-requirements support/golden/qwen38-requirements.txt python support/golden/qwen38.py --model "$FORJA_MODELS/Qwen3.8-27B-4bit" --out "$FORJA_MODELS/Qwen3.8-27B-4bit/forja-fixtures"
```

## Porting notes

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
