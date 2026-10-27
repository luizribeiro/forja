"""Bind production-resolved build snapshots to local source and guest coverage."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from bench_cache import digest


def git(root: Path, *arguments: str) -> bytes:
    """Read only local objects; no network operations."""
    return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                          capture_output=True).stdout


def source_fingerprints(root: Path, commit: str, family: str) -> dict:
    """Unknown Rust/ABI changes remain global; shader compilation is one module."""
    tree = {}
    for entry in git(root, "ls-tree", "-rz", commit).split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            tree[path.decode()] = metadata.split()[-1].decode()
    guest_prefix = f"engines/{family}/"
    guest = {path: blob for path, blob in tree.items() if path.startswith(guest_prefix)
             and not path.startswith((guest_prefix + "profiles/", guest_prefix + "bench/"))
             and not path.endswith("README.md")}
    host = {path: blob for path, blob in tree.items()
            if path.startswith(("crates/", "wit/", "support/guests/", "support/golden-fixtures/"))
            and not path.endswith(".metal")}
    toolchain = {path: blob for path, blob in tree.items()
                 if path in {"Cargo.toml", "Cargo.lock", "flake.nix", "flake.lock", "rust-toolchain.toml"}
                 or path.startswith("nix/")}
    shaders = {path: hashlib.sha256(git(root, "show", f"{commit}:{path}")).hexdigest()
               for path in tree if path.startswith("crates/forja-metal/") and path.endswith(".metal")}
    if not all([guest, host, toolchain, shaders]):
        raise ValueError("source tree does not cover build dependencies")
    return {"guest_source_hash": digest(guest), "host_hash": digest(host),
            "toolchain_hash": digest(toolchain), "kernel_source_hash": digest(shaders)}


def snapshot(build: dict, root: Path, proof: dict | None) -> dict:
    """Keep resolved picks from the binary, never reimplement override precedence."""
    sources = source_fingerprints(root, build["commit"], build["profile_family"])
    catalog = {entry["name"]: {"source_hash": sources["kernel_source_hash"],
                               "properties": entry} for entry in build["catalog"]["variants"]}
    return {**build, **sources, "catalog": catalog, "proof": proof,
            "proof_hash": digest(proof) if proof is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="forja bench-snapshot JSON")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--proof", type=Path, help="guest-source-bound scoreboard reachability")
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    proof = json.loads(args.proof.read_text()) if args.proof else None
    args.json.write_text(json.dumps(snapshot(json.loads(args.build.read_text()),
                                             args.repository, proof), indent=2) + "\n")


if __name__ == "__main__":
    main()
