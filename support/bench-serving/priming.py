"""Bounded requests exercise transport and cache paths before timed cases."""
import http_client
import provenance
import time


def overlap(samples):
    events = [(point, delta) for sample in samples
              for point, delta in zip(sample["request_window"], [1, -1])]
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    return maximum


def case(client, measured):
    """Keep the measured case immutable and bound only the untimed outputs."""
    budget = 32 if measured.get("tools") else 8 if measured["concurrency"] > 1 else 2
    prime = {**measured, "output_tokens": min(budget, measured["output_tokens"])}
    if prime.get("cancel_after_tokens"):
        prime["cancel_after_tokens"] = 1
    result = http_client.run_case(client, prime, 0, 1, raw_samples=True)
    if result["errors"] or result["requests"] != measured["concurrency"]:
        raise ValueError("bounded path prime failed")
    validate_case(prime, result)
    if (prime["sampling"] in {"greedy", "seeded"} and not prime.get("cancel_after_tokens")
            and len(result["output_sha256"]) != 1):
        raise ValueError("bounded prime outputs differ across concurrent requests")
    concurrent = overlap(result["raw_samples"]["requests"])
    if concurrent < min(2, prime["concurrency"]):
        raise ValueError("concurrent priming requests did not overlap")
    return {"policy": "bounded-path-v1", "case": prime,
            "fresh_server_before_measurement": bool(prime.get("cold")),
            "requests": result["requests"], "cached_tokens": result["cached_tokens"], "cache_hits": result["cache_hits"],
            "client_overlap": concurrent, "cancellations": result["cancellations"],
            "seconds": result["measurement_seconds"],
            "server_batch_dispatch_counter": None}


def validate_case(case, result):
    """Refuse cold-cache leakage and missing required cache or cancellation paths."""
    if case.get("cold") and result["cached_tokens"] != 0:
        raise ValueError("cold request reused a prefix")
    if case.get("warm_prefix_tokens") and (result["cached_tokens"] <= 0
            or result.get("cache_hits") != result.get("requests")):
        raise ValueError("required warm-prefix path did not run")
    if case.get("cancel_after_tokens") and result["cancellations"] != result["requests"]:
        raise ValueError("required cancellation path did not run")


def phase_timers(started, phases, **windows):
    """Return exclusive phases and explicitly separate overlapping windows."""
    total = time.perf_counter() - started
    if any(value < 0 for value in phases.values()) or sum(phases.values()) > total + 1e-6:
        raise ValueError("benchmark phase timers overlap")
    return {**phases, "other_seconds": max(0.0, total - sum(phases.values())),
            "total_seconds": total, "overlapping_windows": windows}


def timed_provenance(args, artifacts, command):
    """Include identity hashing in the per-leg runtime budget."""
    started = time.perf_counter()
    identity = provenance.collect(args.repository, args.forja_repository, artifacts,
                                  args.weights, args.resolved_config, command)
    return identity, time.perf_counter() - started
