"""Strict serving-suite loading and expansion."""

import hashlib
import tomllib
from itertools import product
from pathlib import Path

SUITE_KEYS = {
    "schema_version",
    "model",
    "revision",
    "repetitions",
    "warmups",
    "connection_policy",
    "seed",
}
CASE_KEYS = {
    "name",
    "route",
    "prompt_tokens",
    "warm_prefix_tokens",
    "output_tokens",
    "concurrency",
    "sampling",
    "cold",
    "cache_hit",
    "stream",
    "tools",
    "tool_choice",
    "cancel_after_tokens",
}


def load(path: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Load a strict suite and return its stable content hash."""
    data = path.read_bytes()
    document = tomllib.loads(data.decode())
    if set(document) != {"suite", "case"}:
        raise ValueError("suite must contain [suite] and [[case]] only")
    settings = document["suite"]
    if set(settings) != SUITE_KEYS:
        raise ValueError("suite settings have unknown or missing keys")
    if settings["schema_version"] != 1 or settings["connection_policy"] != "reuse":
        raise ValueError("unsupported suite schema or connection policy")
    positive(settings, "repetitions")
    if not isinstance(settings["warmups"], int) or settings["warmups"] < 0:
        raise ValueError("suite.warmups must be nonnegative")
    cases = document["case"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite requires at least one case")
    names = set()
    for case in cases:
        if not {"name", "route", "prompt_tokens", "output_tokens", "concurrency", "sampling"} <= set(case):
            raise ValueError("case is missing required keys")
        if set(case) - CASE_KEYS:
            raise ValueError("case has unknown keys")
        if case["name"] in names:
            raise ValueError("case names must be unique")
        names.add(case["name"])
        if case["route"] not in {"chat", "completions"}:
            raise ValueError("case route is unsupported")
        if case["sampling"] not in {"greedy", "qwen-defaults", "seeded"}:
            raise ValueError("case sampling is unsupported")
        if "tool_choice" in case and (
            not case.get("tools") or case["tool_choice"] not in {"auto", "required"}
        ):
            raise ValueError("case.tool_choice requires tools and must be auto or required")
        positive(case, "output_tokens")
        for key in ("prompt_tokens", "concurrency"):
            values = case[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"case.{key} must be nonempty")
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"case.{key} values must be positive")
    return settings, cases, hashlib.sha256(data).hexdigest()


def expand(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    """Expand prompt/concurrency axes into concrete requests."""
    expanded = []
    for case in cases:
        for prompt_tokens, concurrency in product(case["prompt_tokens"], case["concurrency"]):
            expanded.append({**case, "prompt_tokens": prompt_tokens, "concurrency": concurrency})
    return expanded


def positive(table: dict[str, object], key: str) -> None:
    """Require one non-boolean positive integer."""
    value = table[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
