"""Path-independent provenance for serving benchmark records."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash one artifact without retaining its path."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    """Hash an unordered shard set from its content identities."""
    digests = sorted(bytes.fromhex(sha256_file(path)) for path in paths)
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(digest)
    return combined.hexdigest()


def git_state(repository: Path) -> dict[str, object]:
    """Return commit identity and dirty state for one repository."""
    commit = command(["git", "rev-parse", "HEAD"], repository)
    dirty = bool(command(["git", "status", "--porcelain"], repository, allow_empty=True))
    return {"commit": commit, "dirty": dirty}


def package_version(name: str) -> str | None:
    """Return an installed Python distribution version when present."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def machine() -> dict[str, object]:
    """Describe the benchmark host without embedding local paths."""
    displays = json.loads(command(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"]))
    device = graphics_device(displays)
    return {
        **device,
        "logical_cpu_count": os.cpu_count(),
        "os": f"macOS {platform.mac_ver()[0]}",
    }


def graphics_device(displays: dict[str, object]) -> dict[str, object]:
    """Extract the Metal device identity from system_profiler JSON."""
    for display in displays.get("SPDisplaysDataType", []):
        name = display.get("sppci_model") or display.get("_name")
        cores = display.get("sppci_cores")
        if name and cores:
            return {"device": name, "gpu_core_count": int(cores)}
    raise RuntimeError("system_profiler returned no GPU identity")


def collect(
    repository: Path,
    forja_repository: Path,
    artifacts: dict[str, Path],
    weights: list[Path],
    resolved_config: Path | None,
    server_command: list[str] | None,
) -> dict[str, object]:
    """Collect complete implementation and input provenance."""
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in artifacts.items()}
    hashes["weights_sha256"] = sha256_files(weights) if weights else None
    config = json.loads(resolved_config.read_text()) if resolved_config else None
    return {
        "runner": git_state(repository),
        "forja": git_state(forja_repository),
        **hashes,
        "mlx": package_version("mlx"),
        "mlx_lm": package_version("mlx-lm"),
        "resolved_host_config": config,
        "server_command": server_command,
        **machine(),
    }


def command(arguments: list[str], cwd: Path | None = None, allow_empty: bool = False) -> str:
    """Run a read-only provenance command."""
    value = subprocess.run(arguments, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    if not value and not allow_empty:
        raise RuntimeError(f"{arguments[0]} returned no output")
    return value
