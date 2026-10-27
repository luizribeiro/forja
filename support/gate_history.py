"""Load only recent, verified block-noise history."""
import json
import math
from pathlib import Path
import statistics
import time
from gate_records import verify_artifacts
from gate_stats import METHOD

def history(row: dict, now: float | None = None) -> dict:
    """Only recent, identity-matched, verified block history supplies a budget."""
    prepared = {key: value for key, value in row.items() if key not in {"block_sd", "D"}}
    if not row.get("history"):
        return {**prepared, "history_status": "missing"}
    specification = row["history"]
    if not specification.get("sha256") or not row.get("history_identity"):
        raise ValueError("history requires a pinned SHA256 and identity")
    verify_artifacts({specification["path"]: specification["sha256"]})
    record = json.loads(Path(specification["path"]).read_text())
    if record["identity"] != row["history_identity"] or record.get("method") != METHOD:
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
