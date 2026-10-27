"""Regression tests on recorded profile history; no builds, git history or GPU execution."""

from copy import deepcopy
import json
from pathlib import Path

from gate_select import select

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gate_history"


def history_pair(name):
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    return fixture["rows"], fixture["baseline"], fixture["candidate"]


def test_long_cache_profile_history_selects_only_reachable_decode_ranges():
    rows, baseline, candidate = history_pair("long-cache-43384e5")
    result = select(rows, baseline, candidate)
    assert {r["id"] for r in result["rows"] if r["affected"]} == {"decode-64000", "decode-128000"}


def test_grouped_projection_profile_history_selects_prefill_consumers():
    rows, baseline, candidate = history_pair("grouped-projection-e3c1313")
    affected = {r["id"] for r in select(rows, baseline, candidate)["rows"] if r["affected"]}
    assert affected == {r["id"] for r in rows if not r["id"].startswith("decode-")}
    changed = deepcopy(candidate)
    changed["host_hash"] = "encoding change"
    assert all(r["affected"] for r in select(rows, baseline, changed)["rows"])
