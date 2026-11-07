"""Distribution-free statistics shared by serving benchmark clients."""


def stats(values: list[float]) -> dict[str, object]:
    """Return a median and distribution-free 95% order-statistic interval."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    values = sorted(values)
    count = len(values)
    median = (
        values[count // 2]
        if count % 2
        else (values[count // 2 - 1] + values[count // 2]) / 2
    )
    probability = 0.5**count
    cumulative = probability
    rank = 0
    for index in range(count // 2):
        probability *= (count - index) / (index + 1)
        if cumulative + probability > 0.025:
            break
        cumulative += probability
        rank = index + 1
    return {
        "median": median,
        "p50": median,
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "ci95": [values[rank], values[-rank - 1]],
    }


def percentile(values: list[float], percent: int) -> float:
    """Return one nearest-rank percentile from sorted nonempty values."""
    rank = max(1, (len(values) * percent + 99) // 100)
    return values[rank - 1]


def summary(tokens: int, times: list[float]) -> dict[str, object]:
    """Build one throughput and latency summary."""
    return {
        "tokens_per_second": stats([tokens / elapsed for elapsed in times]),
        "latency_seconds": stats(times),
    }


def pooled_ratio(clusters: list[tuple[float, float]]) -> dict[str, float]:
    """Pool numerator/denominator pairs with a whole-cluster delta-method SE.

    Each pair must represent an independent stream or measurement block. The
    residual sandwich keeps unequal stream lengths without treating tokens as
    independent observations. This SE is descriptive, not a calibrated gate.
    """
    import math

    if len(clusters) < 2:
        raise ValueError("at least two independent clusters are required")
    if any(not math.isfinite(x) or not math.isfinite(y) or x < 0 or y <= 0
           for x, y in clusters):
        raise ValueError("ratios need finite nonnegative numerators and positive denominators")
    numerator = math.fsum(x for x, _ in clusters)
    denominator = math.fsum(y for _, y in clusters)
    ratio = numerator / denominator
    residuals = math.fsum((x - ratio * y) ** 2 for x, y in clusters)
    count = len(clusters)
    se = math.sqrt(count / (count - 1) * residuals) / denominator
    return {"ratio": ratio, "standard_error": se}
