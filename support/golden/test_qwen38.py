"""Fixture binding preserves numerical data and checks both source identities."""
import json
from types import SimpleNamespace

import pytest
from fixture_library import sha256
from qwen38 import bind_grant


def test_binding_retains_source_and_tensor_hashes(tmp_path):
    grant = tmp_path / "model.safetensors"
    grant.write_bytes(b"text grant")
    fixture = tmp_path / "prompt.safetensors"
    header = json.dumps({"prompt_token_ids":{"dtype":"I64","shape":[7],"data_offsets":[0,56]}}).encode()
    fixture.write_bytes(len(header).to_bytes(8,"little") + header + bytes(56))
    original = fixture.read_bytes()
    source = {"file":"index.json", "sha256":"source", "shards":{"shard.safetensors":"weights"}}
    manifest = {"model":source,"prompts":[{"file":fixture.name,"sha256":sha256(fixture)}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    pinned = SimpleNamespace(TEXT_SHA256=sha256(grant),SOURCE_HASHES={"index.json":"source","shard.safetensors":"weights"})
    bind_grant(tmp_path,grant,pinned)
    bound = json.loads(path.read_text())
    assert bound["tensor_source_model"] == source
    assert bound["model"]["sha256"] == pinned.TEXT_SHA256
    assert bound["prompts"][0]["prompt_tokens"] == 7
    assert fixture.read_bytes() == original
    bind_grant(tmp_path,grant,pinned)
    assert json.loads(path.read_text()) == bound
    fixture.write_bytes(original + b"changed")
    with pytest.raises(ValueError,match="fixture hash"):
        bind_grant(tmp_path,grant,pinned)


def test_binding_refuses_an_unpinned_grant(tmp_path):
    grant = tmp_path / "model.safetensors"
    grant.write_bytes(b"changed")
    with pytest.raises(ValueError,match="text grant"):
        bind_grant(tmp_path,grant,SimpleNamespace(TEXT_SHA256="pinned"))
