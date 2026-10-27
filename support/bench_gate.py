"""Run frozen gates with explicit classic controls and fail-closed verdicts."""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
from bench_cache import digest
from gate_history import history
from gate_records import command_sampler, save, verify_artifacts
from gate_runner import run_row
from gate_select import select

def verdict(decisions: list[str]) -> str:
    """Diagnostics, partial measurements and failures cannot become a gating PASS."""
    if not decisions:
        raise ValueError("gate verdict requires at least one row")
    values = set(decisions)
    if values <= {"pass", "unchanged-drift-check"}:
        return "pass"
    if "no change" in values and values <= {"no change", "unchanged-drift-check"}:
        return "no change"
    if "fixed-complete" in values and values <= {"fixed-complete", "unchanged-drift-check"}:
        return "fixed-complete"
    return "not-passed"

def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Accelerated: sequential stopping, dependency selection and valid competitor cache. "
        "Classic: --no-early-stop --reps 30 --all-rows --fresh-competitor. "
        "Native acceptance: forja bench --accept --remeasure. Every mode is recorded."))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--fresh-competitor", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not plan.get("rows") or (not args.no_early_stop and args.reps != 30):
        raise ValueError("require nonempty rows; sequential cap is 30 (--reps changes fixed mode)")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("output directory must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    baseline = json.loads(Path(plan["baseline_snapshot"]).read_text()) if plan.get("baseline_snapshot") else {}
    candidate = json.loads(Path(plan["candidate_snapshot"]).read_text()) if plan.get("candidate_snapshot") else {}
    selection = select(plan["rows"], baseline, candidate, args.all_rows)
    warmups = args.warmups if args.warmups is not None else (3 if args.no_early_stop else 1)
    if warmups < 0 or args.reps <= 0:
        raise ValueError("invalid repetition/warmup counts")
    report = {"schema_version": 1, "plan_sha256": digest(plan), "seed": seed,
              "mode": {"sampling": "fixed" if args.no_early_stop else "sequential",
                       "row_selection": selection["mode"], "reps": args.reps,
                       "competitor": "fresh" if args.fresh_competitor else "cached-if-valid",
                       "warmups_per_launch": warmups},
              "selection": selection, "rows": [], "started": time.time()}
    save(args.out / "plan.json", plan)
    save(args.out / "report.json", report)
    if args.plan_only:
        return
    repository = plan.get("repository", str(Path.cwd()))
    dirty = subprocess.run(["git", "-C", repository, "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout
    if dirty:
        raise ValueError("gate requires a clean source tree")
    verify_artifacts(plan["artifacts"])
    try:
        for index, chosen in enumerate(selection["rows"]):
            row = history(plan["rows"][index])
            directory = args.out / f"row-{index:02}"
            directory.mkdir()
            sampler = command_sampler(row, directory, plan["guard"], warmups, not args.no_early_stop)
            def progress(value, destination=directory):
                save(destination / "progress.json", value)
            result = run_row(row, sampler, seed + index,
                affected=chosen["affected"] or "block_sd" not in row,
                no_early_stop=args.no_early_stop, reps=args.reps, progress=progress)
            report["rows"].append(result)
            if result["runs"] and row.get("history_identity"):
                save(directory / "history.json", {"identity": row["history_identity"],
                    "created_at": time.time(), "method": result["method"], "log_scores": [
                        block["log_score"] for block in result["runs"][0]["blocks"]]})
            save(args.out / "report.json", report)
        if plan.get("competitor_request"):
            command = [sys.executable, str(Path(__file__).with_name("bench_cache_run.py")),
                plan["competitor_request"], "--cache", plan["competitor_cache"],
                "--json", str(args.out / "competitor.json")]
            if args.fresh_competitor:
                command.append("--fresh-competitor")
            subprocess.run([*plan["guard"], *command], check=True)
            report["competitor"] = json.loads((args.out / "competitor.json").read_text())
        verify_artifacts(plan["artifacts"])
        report["ended"] = time.time()
        report["wall_seconds"] = report["ended"] - report["started"]
        report["decision"] = verdict([row["decision"] for row in report["rows"]])
        report["gating_pass"] = report["decision"] == "pass"
        report["diagnostic"] = report["decision"] == "no change"
    except Exception as error:
        report.update(error=str(error), decision="error", ended=time.time())
        raise
    finally:
        save(args.out / "report.json", report)
    if report["decision"] == "not-passed":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
