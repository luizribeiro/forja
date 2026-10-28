"""Load only recent, verified block-noise history."""
import json
import math
from pathlib import Path
import statistics
import time
from gate_records import verify_artifacts
from gate_stats import METHOD

IDENTITY_FIELDS = {"row", "workload_sha256", "harness_protocol", "machine_identity", "os_build", "guard"}


def noise_identity(identity, legacy_workload_sha256=None, legacy_guard=None):
    """Separate reusable noise identity from provenance of the measured builds."""
    if not isinstance(identity, dict):
        raise ValueError("noise identity requires structured workload and protocol fields")
    if "workload_sha256" not in identity:
        if legacy_workload_sha256 is None:
            raise ValueError("legacy noise identity requires an audited workload hash")
        machine = identity["machine_identity"]
        identity = {"row": identity["row"], "workload_sha256": legacy_workload_sha256,
                    "harness_protocol": {"version": identity["protocol"],
                                         "measurement_protocol": identity["measurement_protocol"]},
                    "machine_identity": {"hostname": machine["hostname"], "device": machine["device"],
                                         "description": identity["machine"]}, "os_build": machine["os_build"],
                    "guard": legacy_guard}
    if set(identity) != IDENTITY_FIELDS or any(not identity[k] for k in IDENTITY_FIELDS - {"guard"}):
        raise ValueError("noise identity requires exactly row, workload, protocol, machine, OS build and guard")
    guard = identity["guard"]
    if not isinstance(guard, list) or any(not isinstance(arg, str) or not arg for arg in guard):
        raise ValueError("noise identity guard must be an argv prefix")
    hashed = identity["workload_sha256"]
    if not isinstance(hashed, str) or len(hashed) != 64 or any(c not in "0123456789abcdef" for c in hashed):
        raise ValueError("noise workload hash must be SHA256")
    protocol = identity["harness_protocol"]
    if not isinstance(protocol, dict) or set(protocol) != {"version", "measurement_protocol"}:
        raise ValueError("noise identity requires versioned effective harness protocol")
    if not protocol["version"] or not isinstance(protocol["measurement_protocol"], dict):
        raise ValueError("noise identity requires an effective harness protocol")
    return identity


def history(row: dict, now: float | None = None) -> dict:
    """Only recent, identity-matched, verified block history supplies a budget."""
    prepared = {key: value for key, value in row.items() if key not in {"block_sd", "D"}}
    if row.get("history_identity"):
        prepared["history_identity"] = noise_identity(row["history_identity"])
    if not row.get("history"):
        return {**prepared, "history_status": "missing"}
    specification = row["history"]
    if not specification.get("sha256") or not row.get("history_identity"):
        raise ValueError("history requires a pinned SHA256 and identity")
    verify_artifacts({specification["path"]: specification["sha256"]})
    record = json.loads(Path(specification["path"]).read_text())
    expected = prepared["history_identity"]
    actual = noise_identity(record["identity"], specification.get("legacy_workload_sha256"),
                            specification.get("legacy_guard"))
    if expected["row"] != row["id"] or actual != expected or record.get("method") != METHOD:
        raise ValueError("history identity or method mismatch")
    now = time.time() if now is None else now
    age = specification.get("max_age_seconds", 7 * 86400)
    created = record.get("created_at")
    if not math.isfinite(age) or age <= 0:
        raise ValueError("history max age must be positive and finite")
    if created is None or not math.isfinite(created) or not 0 <= now - created < age:
        return {**prepared, "history_status": "stale-or-undated"}
    scores = record["log_scores"]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("invalid raw block history")
    if len(scores) < 6:
        return {**prepared, "history_status": "insufficient"}
    sd = statistics.stdev(scores)
    if sd == 0:
        return {**prepared, "history_status": "zero-variance"}
    return {**prepared, "block_sd": sd, "D": sd, "history_status": "verified"}
