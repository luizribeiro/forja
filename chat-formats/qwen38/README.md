# Qwen3.8 text chat

A zero-grant chat component for Qwen3.8-27B. It embeds the pinned Qwen template
with SHA-256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
The template retains its upstream Apache-2.0 license.

Format options must be `{}`. Rendering sets `enable_thinking=false` and rejects
requests that enable thinking, tools, image/video content or non-string message
content. Output streams as plain text. Stop tokens are 248046 and 248044.

The rendering fixture is the original template applied to a single `Hello` user
message with `add_generation_prompt=true` and `enable_thinking=false`. The test
checks the complete rendered text, including the empty thinking block; a separate
byte-hash test checks the entire embedded template. Build it with the
[engine profile](../../engines/qwen38/README.md).
