"""Suffix lookup checks against a direct prefix-only oracle."""
import random
import unittest
from prompt_lookup import SuffixLookup


def naive(history, minimum, depth):
    if len(history) < minimum:
        return []
    for start in range(len(history) - minimum - 1, -1, -1):
        if history[start:start + minimum] == history[-minimum:]:
            return history[start + minimum:start + minimum + depth]
    return []


class LookupTests(unittest.TestCase):
    def test_threshold_is_a_minimum_and_latest_match_wins(self):
        history = [1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3]
        self.assertEqual(SuffixLookup(history, 3).proposal(2), [8, 1])
        self.assertEqual(SuffixLookup(history, 4).proposal(2), [])

    def test_streaming_matches_naive_without_future_tokens(self):
        rng = random.Random(1907)
        for minimum in [1, 7, 8, 16, 32, 33]:
            history = []
            lookup = SuffixLookup(history, minimum)
            for _ in range(4097):
                token = rng.randrange(3)
                history.append(token)
                lookup.extend([token])
                self.assertEqual(lookup.proposal(7), naive(history, minimum, 7))

    def test_bad_parameters(self):
        with self.assertRaises(ValueError):
            SuffixLookup([], 0)
        with self.assertRaises(ValueError):
            SuffixLookup([], 1).proposal(-1)
