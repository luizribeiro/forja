"""Dependency-free OpenAI serving benchmark transport."""

import http.client
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from urllib.parse import urlsplit

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_stats import stats  # noqa: E402


def prompt(tokens: int, prefix_tokens: int = 0) -> str:
    """Build deterministic text shared byte-for-byte by both servers."""
    return "".join([" prefix"] * prefix_tokens + [" token"] * tokens).lstrip()


class OpenAIClient:
    """One thread-local persistent HTTP/1.1 connection pool."""

    def __init__(self, base_url: str, model: str, timeout: float = 300.0):
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("server URL must use http")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.base_path = parsed.path.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.local = threading.local()

    def connection(self) -> http.client.HTTPConnection:
        """Return this worker thread's reusable connection."""
        connection = getattr(self.local, "connection", None)
        if connection is None:
            connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            self.local.connection = connection
        return connection

    def reset_connection(self) -> None:
        """Discard this thread's connection after a managed server restart."""
        connection = getattr(self.local, "connection", None)
        if connection is not None:
            connection.close()
        self.local.connection = None

    def request(self, case: dict[str, object], on_first_token: Callable[[], None] | None = None) -> dict[str, object]:
        """Execute one concrete completion and retain only timings and counts."""
        body = request_body(case, self.model)
        route = "/v1/chat/completions" if case["route"] == "chat" else "/v1/completions"
        return self._request(route, body, case.get("cancel_after_tokens", 0), on_first_token)

    def chat(self, messages: list[dict[str, object]], output_tokens: int) -> dict[str, object]:
        """Execute one streaming greedy chat request with an explicit history."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
        }
        return self._request("/v1/chat/completions", body, 0)

    def _request(
        self, route: str, body: dict[str, object], cancel_after_tokens: int,
        on_first_token: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        """Send one prepared OpenAI request and collect its timing sample."""
        started = time.perf_counter()
        try:
            connection = self.connection()
            connection.request(
                "POST",
                self.base_path + route,
                json.dumps(body, separators=(",", ":")),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                message = response.read(4096).decode(errors="replace")
                return failed(started, f"HTTP {response.status}: {message}")
            if body.get("stream"):
                return self.read_stream(response, cancel_after_tokens, started, on_first_token)
            payload = json.loads(response.read())
            ended = time.perf_counter()
            usage = payload.get("usage", {})
            output = response_output(payload)
            return sample(
                started,
                ended,
                ended,
                [],
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                False,
                output,
            )
        except (OSError, ValueError, json.JSONDecodeError, http.client.HTTPException) as error:
            self.local.connection = None
            return failed(started, str(error))

    def read_stream(
        self,
        response: http.client.HTTPResponse,
        cancel_after_tokens: int,
        started: float,
        on_first_token: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        """Read SSE while measuring first-byte and inter-event latency."""
        arrivals = []
        output = []
        usage = {}
        cancelled = False
        limit = cancel_after_tokens
        while line := response.readline():
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            payload = json.loads(data)
            if payload.get("error"):
                response.close()
                self.local.connection = None
                return failed(started, f"SSE error: {payload['error']}")
            if payload.get("usage"):
                usage = payload["usage"]
            choices = payload.get("choices", [])
            if choices and has_content(choices[0]):
                arrivals.append(time.perf_counter())
                output.append(response_output(payload))
                if len(arrivals) == 1 and on_first_token is not None:
                    on_first_token()
                if limit and len(arrivals) >= limit:
                    response.close()
                    self.local.connection = None
                    cancelled = True
                    break
        ended = time.perf_counter()
        if not cancelled:
            response.read()
            response.close()
        first = arrivals[0] if arrivals else ended
        itl = [right - left for left, right in zip(arrivals, arrivals[1:])]
        return sample(
            started,
            first,
            ended,
            itl,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", len(arrivals)),
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            cancelled,
            "".join(output),
        )


def request_body(case: dict[str, object], model: str) -> dict[str, object]:
    """Build one identical OpenAI request for either implementation."""
    text = prompt(case["prompt_tokens"], case.get("warm_prefix_tokens", 0))
    body = {"model": model, "max_tokens": case["output_tokens"], "stream": case.get("stream", False)}
    if case["route"] == "chat":
        body["messages"] = [{"role": "user", "content": text}]
    else:
        body["prompt"] = text
    if case["sampling"] == "greedy":
        body.update({"temperature": 0.0, "top_k": 0, "top_p": 1.0})
    elif case["sampling"] == "seeded":
        body.update({"temperature": 0.7, "top_k": 20, "top_p": 0.8, "seed": 17})
    if case.get("stream"):
        body["stream_options"] = {"include_usage": True}
    if case.get("tools"):
        body["tools"] = [{"type": "function", "function": {"name": "lookup", "description": "Lookup a value", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
        body["tool_choice"] = case.get("tool_choice", "auto")
    return body


def run_case(
    client: OpenAIClient,
    case: dict[str, object],
    warmups: int,
    repetitions: int,
    before_request: Callable[[], None] | None = None,
    after_warmups: Callable[[], None] | None = None,
    raw_samples: bool = False,
) -> dict[str, object]:
    """Run one expanded case with its requested concurrency."""
    if before_request is not None and case["concurrency"] != 1:
        raise ValueError("per-request setup requires concurrency 1")

    def request(_: int) -> dict[str, object]:
        if before_request is not None:
            before_request()
        return client.request(case)

    for _ in range(warmups):
        with ThreadPoolExecutor(max_workers=case["concurrency"]) as executor:
            samples = list(executor.map(request, range(case["concurrency"])))
        errors = [sample["error"] for sample in samples if sample["error"] is not None]
        if errors:
            raise RuntimeError(f"serving warmup failed: {errors}")
    if after_warmups is not None:
        after_warmups()
    count = repetitions * case["concurrency"]
    epoch = time.time()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=case["concurrency"]) as executor:
        samples = list(executor.map(request, range(count)))
    elapsed = time.perf_counter() - started
    good = [value for value in samples if value["error"] is None]
    itl = [value for item in good for value in item["itl_seconds"]]
    result = {
        "case": case,
        "requests": count,
        "requests_per_second": count / elapsed,
        "ttft_seconds": stats([value["ttft_seconds"] for value in good]) if good else None,
        "itl_seconds": stats(itl) if itl else None,
        "e2e_seconds": stats([value["e2e_seconds"] for value in good]) if good else None,
        "prompt_tokens_per_second": sum(value["prompt_tokens"] for value in good) / elapsed,
        "output_tokens_per_second": sum(value["output_tokens"] for value in good) / elapsed,
        "cached_tokens": sum(value["cached_tokens"] for value in good),
        "errors": [value["error"] for value in samples if value["error"] is not None],
        "cancellations": sum(value["cancelled"] for value in samples),
        "output_sha256": sorted(
            {
                hashlib.sha256(value["output"].encode()).hexdigest()
                for value in good
                if not value["cancelled"]
            }
        ),
    }
    if raw_samples:
        result["raw_samples"] = {"measurement_window": [epoch, epoch + elapsed],
                                 "requests": samples}
    return result


def has_content(choice: dict[str, object]) -> bool:
    """Return whether one stream chunk carries generated content."""
    return bool(choice.get("text") or choice.get("delta", {}).get("content") or choice.get("delta", {}).get("tool_calls"))


def response_output(payload: dict[str, object]) -> str:
    """Return a stable representation of generated response content."""
    choices = payload.get("choices", [])
    if not choices:
        return ""
    choice = choices[0]
    message = choice.get("message", {})
    delta = choice.get("delta", {})
    value = choice.get("text") or message.get("content") or delta.get("content")
    if value is not None:
        return value
    tools = message.get("tool_calls") or delta.get("tool_calls") or []
    return json.dumps(tools, sort_keys=True, separators=(",", ":"))


def sample(started, first, ended, itl, prompt_tokens, output_tokens, cached_tokens, cancelled, output):
    """Construct one bounded timing sample."""
    return {"ttft_seconds": first - started, "itl_seconds": itl, "e2e_seconds": ended - started, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens, "cancelled": cancelled, "output": output, "error": None}


def failed(started: float, error: str) -> dict[str, object]:
    """Construct a failed sample without response content."""
    ended = time.perf_counter()
    return {"ttft_seconds": ended - started, "itl_seconds": [], "e2e_seconds": ended - started, "prompt_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cancelled": False, "output": "", "error": error}
