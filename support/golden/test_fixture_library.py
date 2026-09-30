"""Tests for the pure fixture-library operations."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import fixture_library
import pytest


def test_sha256(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"forja\n")

    assert fixture_library.sha256(fixture) == (
        "a2806717b69bd86c324ea44a0754e18251912f259f8e492662a865443ac36a1d"
    )


def test_manifest_base(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fixture_library,
        "versions",
        lambda: {"python": "test", "torch": "test"},
    )
    settings = fixture_library.GenerationSettings(seed=7, generated_tokens=3)

    manifest = fixture_library.manifest_base(
        tmp_path / "model",
        "model-hash",
        [("prompt", "Prompt text")],
        {"tensor": "description"},
        settings,
    )

    assert manifest == {
        "schema_version": 1,
        "model": {
            "directory": "model",
            "file": "model.safetensors",
            "sha256": "model-hash",
        },
        "libraries": {"python": "test", "torch": "test"},
        "generation": {
            "attention": "eager",
            "device": "cpu",
            "dtype": "float32",
            "generated_tokens": 3,
            "seed": 7,
            "tokenization": "raw text, add_special_tokens=False",
        },
        "tensors": {"tensor": "description"},
        "prompts": [{"name": "prompt", "text": "Prompt text"}],
    }


def test_manifest_records_layer_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fixture_library, "versions", lambda: {})
    settings = fixture_library.GenerationSettings(num_hidden_layers=4)

    manifest = fixture_library.manifest_base(tmp_path / "model", "hash", [], {}, settings)

    assert manifest["generation"]["num_hidden_layers"] == 4


def test_generation_settings_reject_nonpositive_layer_limit() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        fixture_library.GenerationSettings(num_hidden_layers=0)


def test_fixtures_are_current_checks_manifest_and_hash(tmp_path: Path) -> None:
    fixture = tmp_path / "prompt.safetensors"
    fixture.write_bytes(b"fixture")
    expected = {"schema_version": 1, "prompts": [{"name": "p", "text": "t"}]}
    actual = {
        "schema_version": 1,
        "prompts": [
            {
                "name": "p",
                "text": "t",
                "file": fixture.name,
                "sha256": fixture_library.sha256(fixture),
                "prompt_tokens": 1,
                "router_logits": {"shape": [2, 1, 4]},
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(actual))

    assert fixture_library.fixtures_are_current(tmp_path, expected)

    fixture.write_bytes(b"changed")
    assert not fixture_library.fixtures_are_current(tmp_path, expected)


def test_fixtures_are_current_rejects_missing_or_invalid_manifest(
    tmp_path: Path,
) -> None:
    assert not fixture_library.fixtures_are_current(tmp_path, {})
    (tmp_path / "manifest.json").write_text("not json")
    assert not fixture_library.fixtures_are_current(tmp_path, {})


def write_model_config(tmp_path: Path, config: dict[str, object]) -> None:
    """Write the only file needed by model-directory validation."""
    (tmp_path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("suffix", [".bin", ".pt", ".pth"])
def test_model_directory_rejects_unsafe_weights(
    tmp_path: Path, suffix: str
) -> None:
    write_model_config(tmp_path, {})
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / f"weights{suffix}").write_bytes(b"")

    with pytest.raises(ValueError, match="unsafe model weight"):
        fixture_library.validate_model_directory(tmp_path)


def test_model_directory_rejects_python(tmp_path: Path) -> None:
    write_model_config(tmp_path, {})
    (tmp_path / "modeling.py").write_text("")

    with pytest.raises(ValueError, match="contains Python code"):
        fixture_library.validate_model_directory(tmp_path)


def test_model_directory_rejects_auto_map(tmp_path: Path) -> None:
    write_model_config(tmp_path, {"auto_map": {"AutoModel": "modeling.Model"}})

    with pytest.raises(ValueError, match="requires remote code"):
        fixture_library.validate_model_directory(tmp_path)


def test_model_directory_rejects_custom_architecture(
    monkeypatch, tmp_path: Path
) -> None:
    write_model_config(tmp_path, {"architectures": ["CustomForCausalLM"]})
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(KnownForCausalLM=object()),
    )

    with pytest.raises(ValueError, match="custom architectures"):
        fixture_library.validate_model_directory(tmp_path)


def test_model_weight_file_validates_shards(monkeypatch, tmp_path: Path) -> None:
    class FakeSafetensors:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        @staticmethod
        def keys():
            return ["weight"]

    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fixture")
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"weight": shard.name},
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    monkeypatch.setitem(
        sys.modules,
        "safetensors",
        SimpleNamespace(safe_open=lambda *args, **kwargs: FakeSafetensors()),
    )

    assert fixture_library.model_weight_file(tmp_path) == index_path

    index["weight_map"]["missing"] = shard.name
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="does not match index"):
        fixture_library.model_weight_file(tmp_path)
