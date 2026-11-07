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


class ReplayTests(unittest.TestCase):
    def test_no_trigger_preserves_plain_rounds(self):
        from prompt_lookup import replay
        result = replay([1, 2, 3], [7, 8, 9, 10], 3, 2)
        self.assertEqual(result['bubble_count'], 0)
        self.assertEqual([r['emitted'] for r in result['rounds']], [1, 1, 1])

    def test_two_bubbles_even_at_terminal_episode(self):
        from prompt_lookup import replay
        result = replay([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 1, 2)
        self.assertEqual(result['episodes'], 1)
        self.assertEqual(result['bubble_count'], 2)
        self.assertEqual(sum(r['emitted'] for r in result['rounds']), 5)
        self.assertEqual(result['rounds'][-1]['drafts'], [5])

    def test_zero_accept_exits_before_retry(self):
        from prompt_lookup import replay
        result = replay([1, 2, 3, 1, 9], [1, 2, 3, 4, 5], 1, 2)
        self.assertEqual(result['rounds'][0]['accepted'], 0)
        self.assertEqual(result['rounds'][1]['drafts'], [])
        self.assertTrue(result['rounds'][1]['exit_step'])

    def test_context_cap_and_disabled_policy(self):
        from prompt_lookup import replay
        for kwargs in [{'max_context': 3}, {'enabled': False}]:
            result = replay([1, 2, 3], [1, 2, 3, 1, 2], 1, 2, **kwargs)
            self.assertEqual(result['episodes'], 0)
