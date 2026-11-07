"""Tests for shared serving benchmark statistics."""

import unittest

from bench_stats import stats, summary


def test_order_statistics_and_throughput_summary() -> None:
    distribution = stats([4.0, 1.0, 3.0, 2.0])
    assert distribution["median"] == 2.5
    assert distribution["p90"] == 4.0
    assert distribution["p99"] == 4.0
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


class PooledRatioTests(unittest.TestCase):
    def test_streams_are_independent_units(self):
        from bench_stats import pooled_ratio
        result = pooled_ratio([(1, 2), (3, 2)])
        self.assertEqual(result, {"ratio": 1.0, "standard_error": 0.5})
        self.assertEqual(result, pooled_ratio([(100, 200), (300, 200)]))

    def test_unequal_lengths_pool_counts(self):
        from bench_stats import pooled_ratio
        result = pooled_ratio([(2, 1), (2, 2)])
        self.assertAlmostEqual(result['ratio'], 4 / 3)
        self.assertAlmostEqual(result['standard_error'], 4 / 9)

    def test_invalid_clusters(self):
        from bench_stats import pooled_ratio
        for clusters in [[], [(1, 1)], [(1, 0), (2, 1)],
                         [(float('nan'), 1), (1, 1)]]:
            with self.assertRaises(ValueError):
                pooled_ratio(clusters)
