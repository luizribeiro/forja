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


def replay(prompt, tokens, minimum_match, depth, max_context=None, enabled=True):
    """Reconstruct a reference stream and mark entry/exit bubbles per episode.

    The first output token is already pending. A zero-accept round, exhausted
    match or context cap ends an episode; the following round runs plain.
    """
    if depth < 1 or (max_context is not None and max_context < 1):
        raise ValueError("depth and context cap must be positive")
    if not tokens:
        return {"rounds": [], "episodes": 0, "bubble_count": 0}
    index = SuffixLookup([*prompt, tokens[0]], minimum_match)
    offset, episodes, active, drain = 0, 0, False, False
    rounds, reconstructed = [], tokens[:1]
    while offset + 1 < len(tokens):
        limit = min(depth, len(tokens) - offset - 2)
        allowed = enabled and (max_context is None or len(index.history) < max_context)
        proposal = index.proposal(limit) if allowed and not drain else []
        if active and not proposal:
            active, drain = False, True
        if proposal and not active:
            episodes += 1
            active = True
        accepted = 0
        for guess, actual in zip(proposal, tokens[offset + 1:]):
            if guess != actual:
                break
            accepted += 1
        emitted = accepted + 1
        rounds.append({"offset": offset, "drafts": proposal, "accepted": accepted,
                       "emitted": emitted, "exit_step": drain})
        reconstructed.extend([*proposal[:accepted], tokens[offset + emitted]])
        index.extend(tokens[offset + 1:offset + emitted + 1])
        offset += emitted
        if drain:
            drain = False
        if proposal and accepted == 0:
            active, drain = False, True
    if reconstructed != tokens:
        raise ValueError("replay failed to reconstruct reference tokens")
    return {"rounds": rounds, "episodes": episodes, "bubble_count": 2 * episodes}
