"""Validate the fixed prompt-lookup workload data without running a model."""
import json
from pathlib import Path


SUITE = Path(__file__).resolve().parents[1] / "bench/suites/prompt-lookup.json"


def test_lookup_workloads_preserve_thresholds_and_separate_long_context():
    suite = json.loads(SUITE.read_text())
    assert suite["source"].startswith("Original forja fixtures")
    assert suite["policy"] == {"min_match": 8, "max_suffix": 32, "max_drafts": 2}
    rows = suite["rows"]
    assert len({row["name"] for row in rows}) == len(rows)
    for row in rows:
        assert (SUITE.parent / row["prompt"]).read_text().strip()
        assert row["group"] == ("long" if row["context_tokens"] == 32512 else "regular")
        edit = row["name"].startswith(("code-edit-", "jsonl-copy-"))
        assert row["minimum_speedup"] == (1.15 if edit else .98)
        assert row["output_tokens"] == (1 if row["name"].startswith("prompt-") else 256)
        assert row["context_tokens"] + row["output_tokens"] <= 32768
    for task in ["code-edit", "jsonl-copy", "code", "chat", "thinking", "jsonl-generate", "c4", "prompt"]:
        assert all(any(row["name"] == f"{task}-{context}" for row in rows)
                   for context in ["short", "4k"])
    assert any(row["temperature"] == .7 and row["seed"] == 42 for row in rows)
    assert all(row["concurrency"] == 4 for row in rows if row["name"].startswith("c4-"))


def test_padding_is_varied_and_reproducible():
    from generate_lookup_padding import render
    corpus = (SUITE.parent / json.loads(SUITE.read_text())["padding"]).read_text()
    assert corpus == render()
    documents = corpus.strip().split("\n\n")
    assert len(documents) == len(set(documents)) == 256
    assert len(corpus) > 300_000
    assert "community repair workshop" not in corpus
