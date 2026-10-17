# Qwen3-Coder XML

This format implements the XML tool protocol used by Qwen3-Coder models. Forja's bundled profile
uses it with `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, a quantized distribution of
Qwen3-Coder-30B-A3B-Instruct.

Generation activates tool parsing at this exact trigger:

```text
<tool_call>
```

A complete call has this shape, including newlines:

```xml
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

Text before the trigger remains assistant text. A required or named `tool_choice` appends the
trigger to the rendered prompt. The parser emits a call and its arguments incrementally, rejects
bytes after the closing tag, and therefore supports one tool call per assistant turn. Forja accepts
either value of `parallel_tool_calls`, but still formats only one call; append its tool result and
start another turn for the next call.

The only supported options value is `{}`. Unknown keys are rejected. As a compatibility quirk,
JSON-encoded OpenAI argument strings in earlier assistant messages are decoded to objects before
the Hugging Face template runs. Tool results are rendered by that template as `<tool_response>`
content in a user turn.

Tool schemas may use objects, arrays, strings, numbers, integers, booleans, required and optional
properties, local `$defs`/`$ref`, and `additionalProperties: false`. Primitive parameter enums are
enforced in the generation grammar. Arrays and objects are admitted as JSON and then checked
recursively by the host against their schema. Parameters are emitted in declaration order.

The fixtures in `tests/fixtures/chat-template.json` were recorded from the hash-pinned model's
Hugging Face chat template and tokenizer. Each case stores the input messages, exact rendered text,
and token IDs. Tests compare rendering byte-for-byte, while pre-push model verification confirms
that the embedded template and generation defaults still match the pinned model files.
