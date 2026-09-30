"""Tests for the pure fixture-library operations."""

import json
from pathlib import Path

import fixture_library


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
