"""The model downloader verifies bytes and leaves changed grants untouched."""

import importlib.util
from pathlib import Path

import pytest


def downloader():
    path = Path(__file__).parents[1] / "engines/smollm2/download.py"
    spec = importlib.util.spec_from_file_location("smollm2_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validates_existing_download_without_network(tmp_path, monkeypatch):
    module = downloader()
    model = tmp_path / "SmolLM2-135M-Instruct"
    model.mkdir()
    source = model / "config.json"
    source.write_bytes(b"pinned")
    monkeypatch.setattr(module, "FILES", {"config.json": module.digest(source)})
    assert module.download(tmp_path) == model
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing to replace"):
        module.download(tmp_path)
    assert source.read_bytes() == b"changed"
