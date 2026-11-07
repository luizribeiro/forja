"""Replay suffix-gated proposals using only the observed token prefix."""


class SuffixLookup:
    """Index the most recent earlier continuation at a fixed match threshold."""

    def __init__(self, history, minimum_match):
        if minimum_match < 1:
            raise ValueError("minimum_match must be positive")
        self.minimum_match = minimum_match
        self.history = []
        self.ends = {}
        self.extend(history)

    def extend(self, tokens):
        """Publish observed tokens, retaining earlier continuation starts."""
        for token in tokens:
            end = len(self.history)
            if end >= self.minimum_match:
                self.ends[tuple(self.history[end - self.minimum_match:end])] = end
            self.history.append(token)

    def proposal(self, depth):
        """Return up to depth known successors of the latest matching suffix."""
        if depth < 0:
            raise ValueError("depth must be nonnegative")
        if len(self.history) < self.minimum_match:
            return []
        end = self.ends.get(tuple(self.history[-self.minimum_match:]))
        return [] if end is None else self.history[end:end + depth]
