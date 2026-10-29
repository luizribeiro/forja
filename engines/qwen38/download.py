#!/usr/bin/env python3
"""Pin source bytes and create a text-only grant without changing any tensor bytes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REPOSITORY = "mlx-community/Qwen3.8-27B-4bit"
SOURCE_HASHES = {
    "model-00001-of-00003.safetensors": "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d",
    "model-00002-of-00003.safetensors": "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670",
    "model-00003-of-00003.safetensors": "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a",
    "model.safetensors.index.json": "13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8",
    "config.json": "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85",
    "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}
TOKENIZER_HASHES = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
}
TEXT_SHA256 = "35735f203613e028e6654d7b4f7a3c07da5e5aad896bd5dd453f1529e11d0367"


def digest(path):
    """Hash a file with bounded memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fetch(root, repository, revision, files):
    """Install verified downloads atomically and refuse changed local bytes."""
    for name, expected in files.items():
        destination = root / name
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError(f"refusing to replace changed model file: {destination}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(f"https://huggingface.co/{repository}/resolve/{revision}/{name}", timeout=60) as source:
                    shutil.copyfileobj(source, output)
            if digest(temporary) != expected:
                raise ValueError(f"download hash mismatch: {name}")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def text_grant(root, shards, expected):
    """Copy only text tensors into a deterministic safetensors file."""
    destination = root / "model.safetensors"
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError("refusing to replace changed text weights")
        return destination
    tensors = {}
    for name in sorted(shards):
        path = root / name
        with path.open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError(f"{path.name}: invalid safetensors header length")
            header = json.loads(source.read(length))
        for key, value in header.items():
            if key.startswith(("language_model.model.", "language_model.lm_head.")):
                if key in tensors:
                    raise ValueError(f"{path.name}: duplicate text tensor {key}")
                start, end = value["data_offsets"]
                if not 0 <= start <= end <= path.stat().st_size - 8 - length:
                    raise ValueError(f"{path.name}: invalid text tensor span for {key}")
                tensors[key] = (path,8+length+start,end-start,value)
    if not tensors:
        raise ValueError("checkpoint has no text tensors")
    offset, header = 0, {}
    for key, (_, _, length, value) in sorted(tensors.items()):
        end = offset + length
        if end >= 1 << 64:
            raise ValueError("text weight size overflow")
        header[key] = {"dtype":value["dtype"],"shape":value["shape"],"data_offsets":[offset,end]}
        offset = end
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as output:
            temporary = Path(output.name)
            output.write(len(encoded).to_bytes(8,"little"))
            output.write(encoded)
            for key in sorted(tensors):
                path, start, remaining, _ = tensors[key]
                with path.open("rb") as source:
                    source.seek(start)
                    while remaining:
                        chunk = source.read(min(remaining,16*1024*1024))
                        if not chunk:
                            raise ValueError(f"{path.name}: truncated text tensor {key}")
                        output.write(chunk)
                        remaining -= len(chunk)
        if expected is not None and digest(temporary) != expected:
            raise ValueError("derived text weight hash mismatch")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def download(root=None):
    """Validate pinned weights, architecture, template and the original tokenizer."""
    root = root or Path(os.environ.get("FORJA_MODELS", Path.home() / ".cache/forja/models"))
    model = root / "Qwen3.8-27B-4bit"
    model.mkdir(parents=True, exist_ok=True)
    fetch(model,REPOSITORY,REVISION,SOURCE_HASHES)
    fetch(model,"Qwen/Qwen3.8-27B",TOKENIZER_REVISION,TOKENIZER_HASHES)
    text_grant(model,[name for name in SOURCE_HASHES if name.endswith(".safetensors")],TEXT_SHA256)
    print(model)
    return model


if __name__ == "__main__":
    download()
