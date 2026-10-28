"""Bounded priming must retain measured payloads and validate actual paths."""
import copy

import pytest

import priming


def test_bounded_prime_preserves_payload_and_checks_overlap(monkeypatch):
    measured = {"concurrency": 4, "output_tokens": 128, "sampling": "greedy",
                "route": "chat", "prompt_tokens": 33, "warm_prefix_tokens": 2048}
    original = copy.deepcopy(measured)
    result = {"errors": [], "requests": 4, "cancellations": 0, "cached_tokens": 64, "cache_hits": 4,
              "output_sha256": ["same"], "measurement_seconds": 0.1,
              "raw_samples": {"requests": [{"request_window": [0, 1]}] * 4}}
    calls = []

    def run(_client, case, warmups, reps, raw_samples):
        calls.append((case, warmups, reps, raw_samples))
        return result

    monkeypatch.setattr(priming.http_client, "run_case", run)
    report = priming.case(None, measured)
    assert measured == original
    assert calls == [(measured | {"output_tokens": 8}, 0, 1, True)]
    assert report["client_overlap"] == 4
    assert report["server_batch_dispatch_counter"] is None
    for field, bad in [("cache_hits", 3), ("output_sha256", ["a", "b"]),
                       ("errors", ["failure"]), ("requests", 3),
                       ("raw_samples", {"requests": [{"request_window": [0, 1]}]})]:
        saved = result[field]
        result[field] = bad
        with pytest.raises(ValueError):
            priming.case(None, measured)
        result[field] = saved


def test_cold_prime_requires_isolation_and_cancellation_is_required(monkeypatch):
    measured = {"cold": True, "concurrency": 1, "output_tokens": 128,
                "sampling": "greedy"}
    result = {"errors": [], "requests": 1, "cancellations": 0, "cached_tokens": 0, "cache_hits": 0,
              "output_sha256": ["same"], "measurement_seconds": 0.1,
              "raw_samples": {"requests": [{"request_window": [0, 1]}]}}
    monkeypatch.setattr(priming.http_client, "run_case", lambda *_args, **_kwargs: result)
    assert priming.case(None, measured)["fresh_server_before_measurement"]
    result["cached_tokens"] = 1
    with pytest.raises(ValueError, match="cold"):
        priming.case(None, measured)
    with pytest.raises(ValueError, match="cold"):
        priming.validate_case(measured, {"cached_tokens": 1})
    with pytest.raises(ValueError, match="warm-prefix"):
        priming.validate_case({"warm_prefix_tokens": 2048}, {"cached_tokens": 0})
    measured = {**measured, "cold": False, "stream": True, "cancel_after_tokens": 7}

    def run(_client, case, *_args, **_kwargs):
        assert case["cancel_after_tokens"] == 1
        assert case["output_tokens"] == 2
        return {"errors": [], "requests": 1, "cancellations": 0}

    monkeypatch.setattr(priming.http_client, "run_case", run)
    with pytest.raises(ValueError, match="cancellation"):
        priming.case(None, measured)
