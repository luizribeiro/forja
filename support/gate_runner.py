"""Sequential gate lifecycle, including drift promotion and one fresh confirmation."""

import math
import random
import statistics

from gate_noise_guard import NoiseGuard
from gate_stats import METHOD, block_score, decide, drift_promotes, projected_power


def run_row(row: dict, sample, seed: int, affected=True, no_early_stop=False,
            reps=30, progress=lambda _: None) -> dict:
    """The sampler persists each original record before returning its timing window."""
    if not isinstance(reps, int) or isinstance(reps, bool) or reps <= 0:
        raise ValueError("fixed repetitions must be positive")
    question, x, y = row.get("question", "non-regression"), row.get("X", .02), row.get("Y", 0.)
    decide([], question, x, y)
    phase_random = {}
    result = {"id": row["id"], "profile": row["profile"], "seed": seed,
              "mode": "fixed" if no_early_stop else "sequential",
              "method": "fixed-abba" if no_early_stop else METHOD,
              "question": question, "X": x, "Y": y,
              "power_projection": projected_power(row.get("block_sd"), question, x, y,
                                                    row.get("alternative", 1.02 if question == "improvement" else 1.)),
              "drift_blocks": [], "runs": [], "confirmed_regression": False,
              "improvement_claim": False, "diagnostic": question == "equivalence",
              "drift_protection": "verified-history" if row.get("D") is not None else "unavailable"}
    expected_outputs = {}

    def block(phase, index):
        rng = phase_random.setdefault(phase, random.Random(f"{seed}:{phase}"))
        order = "ABBA" if rng.getrandbits(1) else "BAAB"
        samples = [sample(side, 1, phase, index, leg) for leg, side in enumerate(order)]
        for item in samples:
            if item.get("output_digests"):
                identity = "both" if row.get("require_equal_outputs") else item["side"]
                expected = expected_outputs.setdefault(identity, item["output_digests"])
                if item["output_digests"] != expected:
                    raise ValueError("gate output digest equality contract failed")
        record = {"orientation": order, "samples": samples, "log_score": block_score(samples)}
        return record

    if no_early_stop:
        records = [sample(side, reps, "fixed", 0, leg) for leg, side in enumerate("ABBA")]
        result.update(decision="fixed-complete", reps_per_leg=reps, records=records)
        progress(result)
        return result

    guard = NoiseGuard(row)

    if not affected:
        result["drift_blocks"] = [block("drift", index) for index in range(5)]
        promoted = drift_promotes([b["log_score"] for b in result["drift_blocks"]], row.get("D"))
        scores = [b["log_score"] for b in result["drift_blocks"]]
        result["noise_guard"] = guard.observe(scores, "drift", minimum=5)
        if promoted:
            guard.collect("drift-check-fired" if row.get("D") is not None else "no-noise-history", "drift", 5)
        result["noise_guard"]["collection_fallbacks"] = list(guard.events)
        promoted = promoted or bool(guard.events)
        result["promoted"] = promoted
        progress(result)
        if not promoted:
            result["decision"] = "unchanged-drift-check"
            return result
    def sequential(phase):
        run = {"phase": phase, "blocks": [], "decision": "continue"}
        result["runs"].append(run)
        first_failure = None
        for index in range(30):
            run["blocks"].append(block(phase, index))
            blocks = run["blocks"]
            baseline = [statistics.mean(math.log(s["score"]) for s in b["samples"]
                                        if s["side"] == "A") for b in blocks]
            budget = row.get("D")
            if budget is not None and (not math.isfinite(budget) or budget <= 0):
                raise ValueError("pass drift budget must be positive and finite")
            alarm = budget is not None and len(baseline) >= 6 and abs(statistics.median(baseline[-3:])
                - statistics.median(baseline[:3])) > budget
            scores = [b["log_score"] for b in blocks]
            result["noise_guard"] = guard.observe(scores, phase, alarm)
            decision = decide(scores, question, x, y, alarm)
            if decision["decision"] != "continue":
                run.setdefault("original_decision", dict(decision))
            if decision["decision"] == "fail" and first_failure is None:
                first_failure = dict(decision)
            if guard.events or budget is None and not result["diagnostic"]:
                result["drift_protection"] = "collection-fallback" if guard.events else "unavailable"
                if index < 29:
                    decision.update(decision="continue", reason="full block collection required")
                elif first_failure is not None:
                    decision = {**first_failure, "collected_blocks": 30}
                elif decision["decision"] in {"pass", "invalidated"}:
                    decision.update(decision="history-collected", unprotected_decision=decision["decision"],
                                    reason="no currently verified block-noise budget")
            run.update(decision=decision["decision"], stopping=decision,
                       pass_drift_budget=budget)
            progress(result)
            if decision["decision"] != "continue":
                return run["decision"]
        return run["decision"]

    primary = sequential("primary")
    result["decision"] = primary
    if primary == "fail" and question != "improvement":
        confirmation = sequential("confirmation")
        result["confirmed_failure"] = confirmation == "fail"
        result["confirmed_regression"] = confirmation == "fail" and question == "non-regression"
        result["decision"] = "confirmed-fail" if confirmation == "fail" else "unconfirmed-fail"
    result["improvement_claim"] = question == "improvement" and primary == "pass"
    if question == "equivalence" and primary == "pass":
        result["decision"] = "no change"
    progress(result)
    return result
