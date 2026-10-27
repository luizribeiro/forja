"""Regenerates gate-history fixtures from local git history (needs a full clone)."""

import json
from pathlib import Path
import tomllib

from bench_cache import digest
from gate_snapshot import git, source_fingerprints

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "qwen3-coder"
PROFILE = "qwen3-coder-30b-a3b-instruct-4bit.metal-apple-m3-ultra.q4"


def history_pair(commit, suffix=""):
    profile_name = PROFILE + suffix
    path = f"engines/{FAMILY}/profiles/{profile_name}.toml"
    snapshots = []
    contexts = [512, 4096, 8000, 16000, 32000, 64000, 128000]
    rows = [{"id": f"decode-{n}", "profile": profile_name} for n in contexts]
    rows += [{"id": f"prefill-{n}", "profile": profile_name}
             for n in [512, 2048, 16384, 64000, 128000]]
    rows += [{"id": name, "profile": profile_name} for name in ["serving", "conversation", "rss-32K"]]
    maximum = tomllib.loads(git(ROOT, "show", f"{commit}^:{path}").decode())["workload"]["max-context"]
    rows = [row for row in rows if not row["id"].startswith(("prefill-", "decode-"))
            or int(row["id"].split("-")[1]) + (128 if row["id"].startswith("decode-") else 0) <= maximum]
    for revision in [commit + "^", commit]:
        profile = tomllib.loads(git(ROOT, "show", f"{revision}:{path}").decode())
        sources = source_fingerprints(ROOT, revision, FAMILY)
        variants = profile.pop("variants")
        sites = sorted(set(variants["fixed"]) | set(variants["rules"]))
        names = set(variants["fixed"].values()) | {
            arm["name"] for rule in variants["rules"].values() for arm in rule["arms"]}
        coverage = {}
        for row in rows:
            name = row["id"]
            if name.startswith("decode-"):
                context = int(name.split("-")[1])
                reachable = ["paged-attention.decode", "moe.down-combine", "sample"]
                bounds = [context - 1, context + 127]
            else:
                reachable = [site for site in sites if "decode" not in site and "batch-" not in site]
                bounds = [0, int(name.split("-")[1]) - 1 if name.startswith("prefill-") else 32767]
            coverage[name] = [{"site": site, "ranges": {"position": bounds, "query": [1, 2048]}}
                              for site in reachable]
        proof = {"guest_source_hash": sources["guest_source_hash"], "sites": sites, "rows": coverage}
        snapshots.append({**sources, "commit": revision, "dirty": False,
            "profile_name": profile_name, "non_pick_profile": profile,
            "build_environment": "profile-history-defaults (fixture)",
            "host_environment": "profile-history-same-host (fixture)",
            "variants": variants, "tunings": [], "host_config": {"replay": True},
            "proof": proof, "proof_hash": digest(proof),
            "catalog": {name: {"source_hash": sources["kernel_source_hash"]} for name in names}})
    return rows, snapshots[0], snapshots[1]


FIXTURES = ROOT / "support" / "fixtures" / "gate_history"
CASES = [("43384e5", ".long-context", "long-cache-43384e5"), ("e3c1313", "", "grouped-projection-e3c1313")]


def main():
    for commit, suffix, name in CASES:
        rows, baseline, candidate = history_pair(commit, suffix)
        fixture = {"source_commit": commit, "profile_suffix": suffix,
                   "rows": rows, "baseline": baseline, "candidate": candidate}
        (FIXTURES / f"{name}.json").write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
