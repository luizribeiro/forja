"""Conservative scoreboard selection from resolved build snapshots and guest proofs."""

import argparse
import json
from pathlib import Path
import traceback

from bench_cache import digest

GLOBAL_KEYS = ("host_hash", "guest_source_hash", "toolchain_hash", "non_pick_profile",
               "host_config", "tunings", "proof_hash", "build_environment")


def pick(variants: dict, site: str, parameters: dict, value: int | None = None) -> str:
    """Resolve an already production-validated selection at an inclusive point."""
    if site in variants["fixed"]:
        if site in variants["rules"]:
            raise ValueError("contradictory fixed and rule selection")
        return variants["fixed"][site]
    rule = variants["rules"][site]
    if rule["parameter"] not in parameters:
        raise ValueError("missing reachable parameter range")
    matches = [arm["name"] for arm in rule["arms"] if arm["lo"] <= value <= arm["hi"]]
    if len(matches) != 1:
        raise ValueError("uncovered or overlapping rule arms")
    return matches[0]


def path_changes(path: dict, baseline: dict, candidate: dict) -> list[str]:
    """Compare every segment cut by either build's rule boundaries."""
    site, ranges = path["site"], path["ranges"]
    rules = [s["variants"].get("rules", {}).get(site) for s in [baseline, candidate]]
    parameters = {rule["parameter"] for rule in rules if rule}
    if len(parameters) > 1:
        raise ValueError("rule parameter changed")
    points = [None]
    if parameters:
        low, high = ranges[next(iter(parameters))]
        if not isinstance(low, int) or not isinstance(high, int) or not 0 <= low <= high < 2**32:
            raise ValueError("invalid reachable range")
        points = {low, high}
        for rule in rules:
            if rule:
                for arm in rule["arms"]:
                    points.update(p for p in [arm["lo"], arm["hi"], arm["hi"] + 1]
                                  if low <= p <= high)
    reasons = []
    for point in points:
        names = [pick(s["variants"], site, ranges, point) for s in [baseline, candidate]]
        if names[0] != names[1]:
            reasons.append(f"{site}@{point}: {names[0]} -> {names[1]}")
        for snapshot, name in zip([baseline, candidate], names):
            if not snapshot["catalog"][name].get("source_hash"):
                raise ValueError("catalog source fingerprint missing")
        if names[0] == names[1] and baseline["catalog"][names[0]] != candidate["catalog"][names[1]]:
            reasons.append(f"{site}@{point}: catalog source or properties changed")
    return sorted(set(reasons))


def select(rows: list[dict], baseline: dict, candidate: dict, all_rows=False) -> dict:
    """Any unknown build dependency or incomplete proof selects every row."""
    if all_rows:
        return {"mode": "all-rows", "rows": [
            {**row, "affected": True, "reasons": ["explicit all-rows"]} for row in rows],
            "selection_errors": []}
    fallback = []
    errors = []
    try:
        for snapshot in [baseline, candidate]:
            if snapshot.get("dirty", True) or not snapshot.get("commit"):
                fallback.append("unknown or dirty build provenance")
            if snapshot["proof"]["guest_source_hash"] != snapshot["guest_source_hash"]:
                fallback.append("guest proof does not match source")
            if digest(snapshot["proof"]) != snapshot["proof_hash"]:
                fallback.append("guest proof fingerprint mismatch")
            if set(snapshot["proof"]["sites"]) != set(snapshot["variants"]["fixed"]) | set(snapshot["variants"]["rules"]):
                fallback.append("guest proof omits resolved sites")
        for key in GLOBAL_KEYS:
            if any(key not in snapshot or snapshot[key] is None or snapshot[key] == ""
                   for snapshot in [baseline, candidate]):
                fallback.append(f"missing {key}")
            elif baseline[key] != candidate[key]:
                fallback.append(f"changed {key}")
        results = {}
        for row in rows:
            if row["id"] in results:
                raise ValueError("duplicate scoreboard row")
            if row["profile"] != baseline["profile_name"] or row["profile"] != candidate["profile_name"]:
                raise ValueError("profile identity does not match scoreboard")
            paths = baseline["proof"]["rows"][row["id"]]
            if paths != candidate["proof"]["rows"][row["id"]]:
                raise ValueError("reachability changed")
            results[row["id"]] = [reason for path in paths
                                  for reason in path_changes(path, baseline, candidate)]
    except Exception as error:
        fallback.append(f"incomplete snapshot or coverage: {error}")
        errors.append({"exception": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
        results = {row["id"]: [] for row in rows}
    return {"mode": "all-rows" if all_rows else "dependency-selection",
            "selection_errors": errors,
            "rows": [{**row, "affected": bool(fallback or results[row["id"]]),
                      "reasons": sorted(set(fallback + results[row["id"]]))} for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("scoreboard", type=Path)
    parser.add_argument("--all-rows", action="store_true", help="classic mode: gate every row")
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    args.json.write_text(json.dumps(select(json.loads(args.scoreboard.read_text()),
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
        args.all_rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
