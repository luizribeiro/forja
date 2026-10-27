"""Unknown dependencies must never silently become unaffected rows."""

from copy import deepcopy

import pytest

from bench_cache import digest
from gate_select import GLOBAL_KEYS, select


def snapshots():
    rows = [{"id": name, "profile": "profile"} for name in ["prefill", "decode-short", "decode-long"]]
    proof = {"guest_source_hash": "guest", "rows": {
        "prefill": [{"site": "prefill", "ranges": {"query": [1, 2048]}}],
        "decode-short": [{"site": "decode", "ranges": {"position": [511, 638]} }],
        "decode-long": [{"site": "decode", "ranges": {"position": [63999, 64126]}}]}}
    snapshot = {**dict.fromkeys(GLOBAL_KEYS, "same"), "commit": "commit", "dirty": False,
        "guest_source_hash": "guest", "proof": proof, "proof_hash": digest(proof),
        "profile_name": "profile", "catalog": {name: {"source_hash": name} for name in ["p", "d", "wide"]},
        "variants": {"fixed": {"prefill": "p"}, "rules": {"decode": {"parameter": "position",
            "arms": [{"lo": 0, "hi": 131071, "name": "d"}]}}}}
    return rows, snapshot, deepcopy(snapshot)


def test_ranges_and_unused_sources_select_only_reachable_rows():
    rows, baseline, candidate = snapshots()
    candidate["variants"]["rules"]["decode"]["arms"] = [
        {"lo": 0, "hi": 63998, "name": "d"}, {"lo": 63999, "hi": 131071, "name": "wide"}]
    assert [r["id"] for r in select(rows, baseline, candidate)["rows"] if r["affected"]] == ["decode-long"]
    candidate = deepcopy(baseline)
    candidate["catalog"]["wide"]["source_hash"] = "unused change"
    assert not any(r["affected"] for r in select(rows, baseline, candidate)["rows"])
    candidate["catalog"]["p"]["source_hash"] = "changed prefill"
    assert [r["affected"] for r in select(rows, baseline, candidate)["rows"]] == [True, False, False]


@pytest.mark.parametrize("field", GLOBAL_KEYS)
def test_every_unknown_global_dependency_gates_all_rows(field):
    rows, baseline, candidate = snapshots()
    candidate[field] = "changed"
    assert all(r["affected"] for r in select(rows, baseline, candidate)["rows"])
    del candidate[field]
    assert all(r["affected"] for r in select(rows, baseline, candidate)["rows"])


def test_missing_coverage_and_explicit_classic_mode_gate_all_rows():
    rows, baseline, candidate = snapshots()
    assert all(r["affected"] for r in select(rows, baseline, candidate, all_rows=True)["rows"])
    del candidate["proof"]["rows"]["decode-short"]
    assert all(r["affected"] for r in select(rows, baseline, candidate)["rows"])


def test_internal_selection_errors_are_visible_and_still_gate_every_row(monkeypatch):
    import gate_select
    rows, baseline, candidate = snapshots()
    def broken(*_args):
        raise RuntimeError("internal resolver failure")
    monkeypatch.setattr(gate_select, "path_changes", broken)
    report = select(rows, baseline, candidate)
    assert all(row["affected"] for row in report["rows"])
    assert report["selection_errors"][0]["exception"] == "RuntimeError"
    assert "internal resolver failure" in report["selection_errors"][0]["traceback"]
