"""Drift probes cannot refresh the age of an original competitor measurement."""

import subprocess
import sys

import pytest

from bench_cache import Cache
from bench_cache_run import file_hashes, measure, metric, run
from test_bench_cache import identity


def test_periodic_drift_then_invalidation(tmp_path):
    cache = Cache(tmp_path, identity(), 100)
    calls = []

    def full():
        calls.append("full")
        return {"score": 100}

    def probe():
        calls.append("probe")
        return {"score": 101}

    assert not run(cache, full, probe, ["/score"], 10, .05, now=0)["competitor_cache"]["reused"]
    assert run(cache, full, probe, ["/score"], 10, .05, now=9)["competitor_cache"]["reused"]
    report = run(cache, full, probe, ["/score"], 10, .05, now=10)
    assert report["competitor_cache"]["measured_at"] == 0
    assert calls == ["full", "probe"]
    report = run(cache, full, lambda: {"score": 80}, ["/score"], 10, .05, now=20)
    assert not report["competitor_cache"]["reused"]
    assert report["competitor_cache"]["drift"]["passed"] is False
    assert calls == ["full", "probe", "full"]
    report = run(cache, full, probe, ["/score"], 10, .05, now=21, fresh=True)
    assert not report["competitor_cache"]["reused"]
    assert calls == ["full", "probe", "full", "full"]


def test_probe_failure_is_not_silently_reused(tmp_path):
    cache = Cache(tmp_path, identity(), 100)
    cache.write({"score": 100}, now=0)
    with pytest.raises(KeyError):
        run(cache, lambda: pytest.fail("must stop"), lambda: {}, ["/score"], 1, .05, now=2)


def test_commands_require_success_and_hash_actual_files(tmp_path):
    source = tmp_path / "source"
    source.write_text("first")
    first = file_hashes([str(source)])
    source.write_text("other")
    assert file_hashes([str(source)]) != first
    record = measure([sys.executable, "-c",
                      "import sys,pathlib; pathlib.Path(sys.argv[1]).write_text('{}')", "{output}"])
    assert record == {}
    with pytest.raises(subprocess.CalledProcessError):
        measure([sys.executable, "-c", "raise SystemExit(7)", "{output}"])
    for value in [True, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            metric({"value": value}, "/value")
