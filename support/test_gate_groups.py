"""Long workloads require a separate, explicit gate decision."""
import pytest

from gate_groups import execution_plan
from gate_select import select
from test_gate_select import snapshots


def row(name, context, command=None):
    adapter = {"command": command or [name, "{output}", "{reps}"]}
    return {"id": name, "workload_max_context": context, "baseline": adapter, "candidate": adapter}


def selection(rows, proven=()):
    return {"rows": [{**r, "affected": True, "source_bound_affected": r["id"] in proven} for r in rows]}


@pytest.mark.parametrize("name,context", [
    ("decode32k", 32768), ("decode64k", 65536), ("decode128k", 131072),
    ("prefill64k", 65536), ("prefill128k", 131072)])
def test_long_rows_are_visible_but_excluded_from_regular_plan(name, context):
    rows = [row("short", 32767), row(name, context)]
    result = execution_plan({"rows": rows}, selection(rows))
    assert [r["id"] for r in result["rows"]] == ["short"]
    assert result["skipped_rows"][0]["id"] == name
    assert result["gate_groups"][1]["status"] == "skipped-with-reason"
    assert result["gate_groups"][1]["reason"]


def test_every_metric_sharing_a_long_workload_is_long_even_if_its_own_axis_is_short():
    command = ["forja", "bench", "--pp", "16384", "--tg", "128",
               "--set", "bench.contexts=[32768, 65536]", "--set", "bench.warmups={warmups}"]
    rows = [row("prefill16k", 16384, command), row("batch1", 2048, command)]
    result = execution_plan({"rows": rows}, selection(rows))
    assert not result["rows"]
    assert [r["gate_group"] for r in result["skipped_rows"]] == ["long", "long"]
    assert all(r["workload_max_context"] >= 65536 for r in result["skipped_rows"])


def test_source_bound_affected_row_pulls_in_separate_long_group():
    rows, baseline, candidate = snapshots()
    for r in rows:
        r.update(row(r["id"], 65536 if r["id"] == "decode-long" else 2048))
    candidate["variants"]["rules"]["decode"]["arms"] = [
        {"lo": 0, "hi": 63998, "name": "d"}, {"lo": 63999, "hi": 131071, "name": "wide"}]
    chosen = select(rows, baseline, candidate)
    result = execution_plan({"rows": rows}, chosen, "long")
    assert [r["id"] for r in result["rows"]] == ["decode-long"]
    assert result["gate_groups"][1]["required_by_proof"]
    assert result["gate_groups"][1]["source_bound_affected_rows"] == ["decode-long"]
    candidate["host_hash"] = "unknown change"
    result = execution_plan({"rows": rows}, select(rows, baseline, candidate), "long")
    assert not result["rows"] and result["gate_groups"][1]["status"] == "skipped-with-reason"


def test_unknown_or_all_rows_selection_cannot_enable_long_without_explicit_reason():
    rows = [row("long", 32768)]
    assert not execution_plan({"rows": rows}, selection(rows), "long")["rows"]
    result = execution_plan({"rows": rows}, selection(rows), "long", "periodic coverage")
    assert len(result["rows"]) == 1
    assert result["gate_groups"][1]["manual_or_periodic_reason"] == "periodic coverage"


@pytest.mark.parametrize("ceiling", [None, True, 0, -1, "32768"])
def test_missing_or_invalid_audited_ceiling_is_refused(ceiling):
    rows = [row("unknown", ceiling)]
    with pytest.raises(ValueError, match="workload_max_context"):
        execution_plan({"rows": rows}, selection(rows))
