"""Separate long-context workloads from regular gate decisions."""
import tomllib

from gate_records import session_key

LONG_CONTEXT = 32768


def workload_context(row):
    """Require an audited whole-workload ceiling and include explicit native axes."""
    ceiling = row.get("workload_max_context")
    if type(ceiling) is not int or ceiling < 1:
        raise ValueError("row requires positive workload_max_context for the entire command")
    for side in ["baseline", "candidate"]:
        command = row[side]["command"]
        if len(command) < 2 or command[1] != "bench":
            continue
        options = {}
        for index, arg in enumerate(command[:-1]):
            if arg == "--set" and command[index + 1].partition("=")[0].strip() in {
                    "bench.contexts", "bench.decode_prefill", "bench.pp", "bench.tg"}:
                for section, values in tomllib.loads(command[index + 1]).items():
                    if section == "bench":
                        options.update(values)
        for flag, name in [("--pp", "pp"), ("--tg", "tg")]:
            if flag in command:
                options[name] = int(command[command.index(flag) + 1])
        steps = options.get("tg", 0)
        ceiling = max(ceiling, options.get("pp", 0), options.get("decode_prefill", 0) + steps + 1,
                      *(context + steps for context in options.get("contexts", [])))
    return ceiling


def execution_plan(plan, selection, group="regular", long_reason=None):
    """Select exactly one gate group; preserve every excluded row and its reason."""
    if group not in {"regular", "long"} or long_reason is not None and (
            group != "long" or not long_reason.strip()):
        raise ValueError("long_reason requires a separate long gate and a nonempty reason")
    ceilings = {}
    for row in plan["rows"]:
        key = session_key(row)
        ceilings[key] = max(ceilings.get(key, 0), workload_context(row))
    rows = [{**row, "workload_max_context": ceilings[session_key(row)],
             "gate_group": "long" if ceilings[session_key(row)] >= LONG_CONTEXT else "regular"}
            for row in plan["rows"]]
    proven = {row["id"] for row in selection["rows"] if row.get("affected") and row.get("source_bound_affected")}
    summaries = []
    for name in ["regular", "long"]:
        ids = [row["id"] for row in rows if row["gate_group"] == name]
        affected = sorted(set(ids) & proven)
        eligible = name == "regular" or bool(affected or long_reason)
        selected = name == group and eligible and bool(ids)
        reason = ("requested gate group" if selected else "no rows" if not ids else
                  "separate long gate; excluded from regular decisions" if name == "long" and group == "regular" else
                  "no source-bound affected long row; explicit manual or periodic reason required" if not eligible else
                  "different gate group requested")
        summaries.append({"name": name, "status": "selected" if selected else "skipped-with-reason",
                          "reason": reason, "rows": ids, "source_bound_affected_rows": affected,
                          "required_by_proof": bool(affected), "manual_or_periodic_reason": long_reason if name == "long" else None})
    selected = {item["name"] for item in summaries if item["status"] == "selected"}
    skipped = [{**row, "skip_reason": next(item["reason"] for item in summaries
                 if item["name"] == row["gate_group"])} for row in rows if row["gate_group"] not in selected]
    return {**plan, "rows": [row for row in rows if row["gate_group"] in selected],
            "gate_group": group, "gate_groups": summaries, "skipped_rows": skipped}
