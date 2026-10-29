"""The custom chat component embeds the pinned template without rewriting bytes."""
import hashlib
import json
from pathlib import Path


def test_pinned_template_bytes():
    source = (Path(__file__).parents[1] / "chat-formats/qwen38/src/template.rs").read_text()
    literal = source.split("pub const TEMPLATE: &str = ",1)[1].rstrip().removesuffix(";")
    template = json.loads(literal).encode()
    assert hashlib.sha256(template).hexdigest() == "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
