"""Check reusable noise against observed independent blocks before trusting a PASS."""
import math
import statistics


class NoiseGuard:
    """A history departure irreversibly switches this row to full collection."""

    def __init__(self, row):
        self.stored = row.get("block_sd")
        self.factor = row.get("history_sd_factor", 2.)
        if not isinstance(self.factor, (int, float)) or not math.isfinite(self.factor) or self.factor <= 1:
            raise ValueError("history SD factor must be finite and greater than one")
        if self.stored is not None and (not math.isfinite(self.stored) or self.stored <= 0):
            raise ValueError("stored block SD must be positive and finite")
        self.events = []

    def collect(self, reason, phase, blocks, **details):
        """Retain the first occurrence of each fallback reason, across confirmations."""
        if not any(event["reason"] == reason for event in self.events):
            self.events.append({"reason": reason, "phase": phase, "blocks": blocks, **details})

    def observe(self, scores, phase, drift_alarm=False, minimum=6):
        """Compare SD in both directions; the guard never replaces sign-test evidence."""
        observed = statistics.stdev(scores) if len(scores) >= 2 else None
        if observed is not None and not math.isfinite(observed):
            raise ValueError("observed block SD must be finite")
        if drift_alarm:
            self.collect("baseline-drift", phase, len(scores))
        if self.stored is not None and len(scores) >= minimum and (
                observed > self.stored * self.factor or observed < self.stored / self.factor):
            self.collect("block-sd-departure", phase, len(scores), observed_sd=observed,
                         stored_sd=self.stored, factor=self.factor)
        return {"stored_block_sd": self.stored, "observed_block_sd": observed,
                "sd_factor": self.factor, "collection_fallbacks": list(self.events)}
