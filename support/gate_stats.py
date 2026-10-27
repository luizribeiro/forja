"""Calibrated, distribution-free sequential decisions on randomized four-leg blocks."""

from fractions import Fraction
import math
import statistics

RANKS = {6: 1, 10: 1, 16: 2, 20: 3, 25: 7, 30: 9}
METHOD = "randomized-four-leg-calibrated-sign-v1"


def log_tau(question: str, x: float, y: float) -> float:
    """Use the same rounded ratio boundary for decisions and power projections."""
    return math.log(1 + y) if question == "improvement" else math.log(1 - x)


def operating_characteristics(probability=Fraction(1, 2)) -> dict:
    """Enumerate first-crossing paths, exactly when probability is a Fraction."""
    if not 0 <= probability <= 1:
        raise ValueError("sign probability must be in [0,1]")
    states = {0: 1}
    passed = failed = expected = 0
    for n in range(1, 31):
        following = {}
        for count, mass in states.items():
            following[count] = following.get(count, 0) + mass * (1 - probability)
            following[count + 1] = following.get(count + 1, 0) + mass * probability
        states = following
        k = RANKS.get(n, 0)
        for count, mass in list(states.items()):
            if count >= n - k + 1:
                passed += mass
            elif count <= k - 1:
                failed += mass
            else:
                continue
            expected += n * mass
            del states[count]
    remaining = sum(states.values())
    return {"pass": passed, "fail": failed, "inconclusive": remaining,
            "expected_blocks": expected + 30 * remaining}


def block_score(samples: list[dict]) -> float:
    """Interpolate baseline logs at rep midpoints, including BAAB extrapolation."""
    if len(samples) != 4 or "".join(s["side"] for s in samples) not in {"ABBA", "BAAB"}:
        raise ValueError("block must be ABBA or BAAB")
    previous = -math.inf
    for sample in samples:
        start, end, score = (sample[k] for k in ["start", "end", "score"])
        if (not all(math.isfinite(v) for v in [start, end, score])
                or start < previous or end <= start or score <= 0):
            raise ValueError("invalid or overlapping measurement windows/scores")
        previous = end
    def interpolate_difference(side):
        left, right = [(s["start"] / 2 + s["end"] / 2, math.log(s["score"]))
                       for s in samples if s["side"] == side]
        differences = []
        for sample in samples:
            if sample["side"] != side:
                midpoint = sample["start"] / 2 + sample["end"] / 2
                interpolated = left[1] + (right[1] - left[1]) * (midpoint - left[0]) / (right[0] - left[0])
                differences.append(math.log(sample["score"]) - interpolated)
        return statistics.mean(differences)

    return (interpolate_difference("A") - interpolate_difference("B")) / 2


def decide(scores: list[float], question: str, x: float = .02, y: float = 0,
           drift_alarm: bool = False) -> dict:
    """Cap is inconclusive; drift can void a PASS but can never erase a FAIL."""
    if (question not in {"non-regression", "improvement", "equivalence"}
            or not math.isfinite(x) or not 0 < x < 1 or not math.isfinite(y) or y < 0
            or len(scores) > 30 or any(not math.isfinite(s) for s in scores)):
        raise ValueError("invalid sequential question, thresholds or scores")
    n = len(scores)
    result = {"method": METHOD, "blocks": n, "question": question, "X": x, "Y": y,
              "decision": "continue", "drift_alarm": drift_alarm,
              "diagnostic": question == "equivalence"}
    if n not in RANKS:
        return result
    values = sorted(scores)
    k = RANKS[n]
    low, high = values[k - 1], values[n - k]
    result.update(rank=k, log_interval=[low, high])
    lower = log_tau(question, x, y)
    if question == "equivalence":
        upper = math.log(1 + x)
        if low > lower and high < upper:
            result["decision"] = "pass"
        elif high < lower or low > upper:
            result["decision"] = "fail"
    elif low > lower:
        result["decision"] = "pass"
    elif high < lower:
        result["decision"] = "fail"
    if result["decision"] == "continue" and n == 30:
        result["decision"] = "inconclusive"
    if result["decision"] == "pass" and drift_alarm:
        result["decision"] = "invalidated"
    return result


def projected_power(sd: float | None, question: str, x: float, y: float,
                    alternative: float) -> dict | None:
    """Gaussian sign-power diagnostic; coverage does not assume Gaussian noise."""
    if sd is None:
        return None
    if not math.isfinite(sd) or sd <= 0 or not math.isfinite(alternative) or alternative <= 0:
        raise ValueError("power requires positive block SD and alternative")
    if question == "equivalence":
        return None
    probability = statistics.NormalDist().cdf((math.log(alternative) - log_tau(question, x, y)) / sd)
    return {key: float(value) for key, value in operating_characteristics(probability).items()}


def drift_promotes(scores: list[float], budget: float | None) -> bool:
    """Five blocks without comparable block-noise history always promote."""
    if len(scores) != 5 or any(not math.isfinite(s) for s in scores):
        raise ValueError("drift check requires five finite block scores")
    if budget is None:
        return True
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("drift budget must be positive and finite")
    return statistics.median(scores) < -budget
