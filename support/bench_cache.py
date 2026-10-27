"""Content-addressed competitor records with explicit freshness provenance."""

import hashlib
import json
import math
import os
from pathlib import Path
import time


def digest(value: object) -> str:
    """Hash canonical JSON, rejecting non-finite numbers."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


class Cache:
    """Keep immutable measurements separate from reuse and drift metadata."""

    def __init__(self, directory: Path, identity: dict, max_age: float):
        required = {"competitor_version", "effective_config", "model_hash",
                    "workload_hash", "tokens_hash", "machine", "os"}
        if set(identity) != required or any(not value for value in identity.values()):
            raise ValueError(f"identity must contain nonempty {sorted(required)}")
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max age must be positive and finite")
        self.identity = identity
        self.key = digest(identity)
        self.path = directory / f"{self.key}.json"
        self.max_age = max_age

    def read(self, now: float | None = None) -> dict | None:
        """Treat missing, corrupt, future-dated or expired records as misses."""
        now = time.time() if now is None else now
        try:
            entry = json.loads(self.path.read_text())
            age = now - entry["measured_at"]
            if (isinstance(entry["record"], dict)
                    and entry["identity"] == self.identity and 0 <= age < self.max_age
                    and entry["measured_at"] <= entry["drift_checked_at"] <= now
                    and entry["record_hash"] == digest(entry["record"])):
                return entry
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def write(self, record: dict, now: float | None = None) -> dict:
        """Atomically replace a measurement after a successful fresh run."""
        now = time.time() if now is None else now
        entry = {"identity": self.identity, "record": record,
                 "record_hash": digest(record), "measured_at": now,
                 "drift_checked_at": now}
        self.save(entry)
        return entry

    def save(self, entry: dict) -> None:
        """Publish only complete cache envelopes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(entry, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def report(self, entry: dict, reused: bool, now: float | None = None) -> dict:
        """Label reuse without changing original measurement provenance."""
        return {**entry["record"], "competitor_cache": {
            "key": self.key, "reused": reused, "identity": self.identity,
            "measured_at": entry["measured_at"],
            "age_seconds": (time.time() if now is None else now) - entry["measured_at"],
            "record_sha256": entry["record_hash"],
            "drift_checked_at": entry["drift_checked_at"],
            "drift": entry.get("drift"),
        }}
