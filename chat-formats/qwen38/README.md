# Qwen3.8 text chat

A zero-grant chat component for Qwen3.8-27B. It embeds the pinned Qwen template
with SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
The template retains its upstream Apache-2.0 license.

Format options must be `{}`. Thinking is enabled by default with effort `xhigh`;
`enable_thinking=false`, `reasoning_effort=medium|low`, and
`preserve_thinking=false` select the pinned template's alternatives. Image/video
input is rejected. Stop tokens are 248046 and 248044.

The fixtures were recorded on CPU with Transformers 5.17.0 from the pinned model's
original `chat_template.jinja`, using `AutoTokenizer.apply_chat_template` with
`tokenize=False`. OpenAI JSON argument strings are decoded into mappings before
rendering; insertion order is retained. Regenerate with:

```sh
uv run --offline --no-project --python 3.12 \
  --with-requirements support/golden/qwen38-requirements.txt \
  python support/golden/qwen38-chat.py
```

`FORJA_MODELS` must point to the downloaded models. `tests/fixtures/reference.json`
records the revision, template hash and reference versions. Build the component
with the [engine profile](../../engines/qwen38/README.md).
