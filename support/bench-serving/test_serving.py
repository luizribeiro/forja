"""Tests for the serving benchmark client."""

import tempfile
from pathlib import Path

import suite


def test_committed_suite_covers_serving_axes() -> None:
    path = Path(__file__).resolve().parents[2] / "bench/suites/serving.toml"
    settings, cases, digest = suite.load(path)
    expanded = suite.expand(cases)
    assert settings["connection_policy"] == "reuse"
    assert len(digest) == 64
    assert {case["route"] for case in cases} == {"chat", "completions"}
    assert {value for case in expanded for value in [case["concurrency"]]} == {1, 2, 4, 8}
    assert {value for case in expanded for value in [case["prompt_tokens"]]} >= {1, 7, 33, 512, 4097}
    assert any(case.get("stream") for case in cases)
    assert any(case.get("tools") for case in cases)
    assert any(case.get("cancel_after_tokens") for case in cases)
    assert any(case.get("cache_hit") for case in cases)


def test_suite_rejects_unknown_case_keys() -> None:
    source = (Path(__file__).resolve().parents[2] / "bench/suites/serving.toml").read_text()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "suite.toml"
        path.write_text(source.replace('name = "cold-pp512-tg128"', 'name = "cold-pp512-tg128"\nsecret = true'))
        try:
            suite.load(path)
        except ValueError as error:
            assert "unknown" in str(error)
        else:
            raise AssertionError("unknown suite key was accepted")
