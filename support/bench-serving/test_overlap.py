"""Checks that the interference workload actually joins active decodes."""

import threading

import overlap
import pytest


class Client:
    def __init__(self, *, cached=0, error=None):
        self.lock = threading.Lock()
        self.decoding = 0
        self.prefill_started = threading.Event()
        self.cached = cached
        self.error = error

    def request(self, case, on_first_token=None):
        if on_first_token:
            with self.lock:
                self.decoding += 1
            on_first_token()
            assert self.prefill_started.wait(timeout=1)
        else:
            assert self.decoding == 4
            self.prefill_started.set()
        return {"error": self.error, "cached_tokens": self.cached,
                "itl_seconds": [0.01, 0.02]}


def test_long_prompt_joins_after_all_decodes_start():
    samples, _ = overlap.run_pair(Client(), {"concurrency": 4}, {})
    assert len(samples) == 4


@pytest.mark.parametrize("options", [{"cached": 1}, {"error": "failed"}])
def test_invalid_overlap_is_refused(options):
    with pytest.raises(RuntimeError):
        overlap.run_pair(Client(**options), {"concurrency": 4}, {})
