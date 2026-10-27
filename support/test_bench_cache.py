"""Competitor reuse must never hide changed inputs or measurement age."""

import json
from pathlib import Path

import pytest

from bench_cache import Cache


def identity():
    return {key: {"value": key} for key in (
        "competitor_version", "effective_config", "model_hash", "workload_hash",
        "tokens_hash", "machine", "os",
    )}


def test_reuse_preserves_original_provenance(tmp_path: Path):
    cache = Cache(tmp_path, identity(), 10)
    record = {"provenance": {"commit": "original"}, "results": [1]}
    cache.write(record, now=100)
    entry = cache.read(now=105)
    report = cache.report(entry, reused=True, now=105)
    assert report["provenance"] == record["provenance"]
    assert report["competitor_cache"]["age_seconds"] == 5
    assert report["competitor_cache"]["reused"]
    assert cache.read(now=110) is None
    assert cache.read(now=99) is None
    entry["drift_checked_at"] = 106
    cache.save(entry)
    assert cache.read(now=105) is None


@pytest.mark.parametrize("field", list(identity()))
def test_every_identity_mismatch_misses(tmp_path: Path, field: str):
    Cache(tmp_path, identity(), 10).write({"results": [1]}, now=100)
    changed = {**identity(), field: "changed"}
    assert Cache(tmp_path, changed, 10).read(now=101) is None


def test_corruption_and_incomplete_identity_refuse(tmp_path: Path):
    cache = Cache(tmp_path, identity(), 10)
    cache.write({"results": [1]}, now=100)
    entry = json.loads(cache.path.read_text())
    entry["record"]["results"] = [2]
    cache.path.write_text(json.dumps(entry))
    assert cache.read(now=101) is None
    with pytest.raises(ValueError):
        Cache(tmp_path, {"model_hash": "only"}, 10)
    for age in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            Cache(tmp_path, identity(), age)
