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
