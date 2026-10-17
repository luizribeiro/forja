# Chat formats

A chat format is a zero-import WebAssembly guest that adapts the OpenAI chat API to a model's
wire format. It renders messages with the model's Hugging Face template, builds a grammar for a
tool request, and incrementally parses generated tool calls. The trusted host supplies bounded
JSON strings and receives owned strings and structured parser events; the guest has no host
imports.

## Selecting a format

An engine profile can build a supported format from model files:

```toml
[chat]
tool-format = "qwen3-coder-xml"
tool-format-options = {}
```

`tool-format-options` holds explicit, format-specific compatibility quirks. Unknown options are
rejected. Alternatively, `component = "my-format"` builds `chat-formats/my-format` as a custom
component; a profile must select either `tool-format` or `component`, not both.

## Adding or adapting a format

Add a crate under `chat-formats/<name>` that implements `forja_chat_format::ToolFormat`. Cover its
render normalization, activation trigger, incremental parser, grammar, and malformed inputs in
tests, then register its feature in `forja-chat-format-component` and its option validation in the
`forja` builder.

Add a typed option when models share the protocol but require a small compatibility quirk. Keep the
option profile-controlled and reject unknown keys. For a derived variant, reuse the common parser
and grammar where possible, but give behavior with a different wire contract its own format name
and tests. A fully independent implementation can use the profile's `component` escape hatch.

## Embedded model data

Profiles using `tool-format` pin `chat_template.jinja` and `generation_config.json` with
`model.chat-template-sha256` and `model.generation-config-sha256`. `forja build` reads those files
from the selected model directory, verifies both hashes, and embeds the template, stop-token IDs,
sampling defaults, and serialized format options in the companion chat-format component. Serving
therefore does not read mutable model metadata.

See each format's README for its wire protocol, supported schemas, and compatibility notes.
