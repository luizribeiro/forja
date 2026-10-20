"""Tests for shared serving benchmark statistics."""

from bench_stats import stats, summary


def test_order_statistics_and_throughput_summary() -> None:
    assert stats([4.0, 1.0, 3.0, 2.0])["median"] == 2.5
    result = summary(8, [2.0, 1.0])
    assert result["tokens_per_second"]["median"] == 6.0
    assert result["latency_seconds"]["median"] == 1.5


def test_empty_samples_are_rejected() -> None:
    try:
        stats([])
    except ValueError as error:
        assert str(error) == "cannot summarize an empty sample"
    else:
        raise AssertionError("empty sample was accepted")
