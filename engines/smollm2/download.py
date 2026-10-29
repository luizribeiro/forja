#!/usr/bin/env python3
"""Download the pinned Apache-2.0 SmolLM2 model without executing remote code."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
REPOSITORY = "HuggingFaceTB/SmolLM2-135M-Instruct"
FILES = {
    "README.md": "4f97533ad95b1b2fea15fbc075c01b94578ebdd7c8138888fa43fa3abd530dc4",
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}


def digest(path):
    """Hash a file without retaining its contents in memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(root=None):
    """Validate existing files and atomically install verified downloads."""
    root = root or Path(os.environ.get("FORJA_MODELS", Path.home() / ".cache/forja/models"))
    model = root / "SmolLM2-135M-Instruct"
    model.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = model / name
        if destination.exists():
            if digest(destination) != expected:
                raise ValueError(f"refusing to replace changed model file: {destination}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=model, delete=False) as output:
                temporary = Path(output.name)
                url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
                with urllib.request.urlopen(url, timeout=60) as source:
                    shutil.copyfileobj(source, output)
            if digest(temporary) != expected:
                raise ValueError(f"download hash mismatch: {name}")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(model)
    return model


if __name__ == "__main__":
    download()
