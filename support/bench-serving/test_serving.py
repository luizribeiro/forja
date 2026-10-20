"""Tests for the serving benchmark client."""

import argparse
import importlib.util
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import http_client
import provenance
import serving_bench
import suite


def comparison_module():
    """Load the local compare.py without colliding with the MLX helper."""
    path = Path(__file__).with_name("compare.py")
    spec = importlib.util.spec_from_file_location("serving_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        if request.get("stream"):
            events = [
                {"choices": [{"delta": {"content": "a"}}]},
                {"choices": [{"delta": {"content": "b"}}]},
                {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}}},
            ]
            body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            body = json.dumps({"choices": [{"text": "ab"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 3}}}).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def test_committed_suite_covers_serving_axes() -> None:
    path = Path(__file__).resolve().parents[2] / "bench/suites/serving.toml"
    settings, cases, digest = suite.load(path)
    expanded = suite.expand(cases)
    assert settings["connection_policy"] == "reuse"
    assert len(digest) == 64
    assert {case["route"] for case in cases} == {"chat", "completions"}
    assert {value for case in expanded for value in [case["concurrency"]]} == {1, 2, 4, 8}
    assert {value for case in expanded for value in [case["prompt_tokens"]]} >= {1, 7, 33, 512, 4097}
    assert any(case.get("stream") for case in cases)
    assert any(case.get("tools") for case in cases)
    assert any(case.get("cancel_after_tokens") for case in cases)
    assert any(case.get("cache_hit") for case in cases)


def test_suite_rejects_unknown_case_keys() -> None:
    source = (Path(__file__).resolve().parents[2] / "bench/suites/serving.toml").read_text()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "suite.toml"
        path.write_text(source.replace('name = "cold-pp512-tg128"', 'name = "cold-pp512-tg128"\nsecret = true'))
        try:
            suite.load(path)
        except ValueError as error:
            assert "unknown" in str(error)
        else:
            raise AssertionError("unknown suite key was accepted")


def test_http_client_reuses_openai_transport_and_measures_sse() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client = http_client.OpenAIClient(f"http://127.0.0.1:{server.server_port}", "model")
        case = {"name": "fake", "route": "completions", "prompt_tokens": 7, "output_tokens": 2, "concurrency": 2, "sampling": "greedy"}
        result = http_client.run_case(client, case, 1, 2)
        assert result["requests"] == 4
        assert not result["errors"]
        assert result["cached_tokens"] == 12
        streamed = client.request({**case, "stream": True})
        assert streamed["output_tokens"] == 2
        assert len(streamed["itl_seconds"]) == 1
        cancelled = client.request({**case, "stream": True, "cancel_after_tokens": 1})
        assert cancelled["cancelled"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_provenance_hashes_content_without_paths() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        combined = provenance.sha256_files([first, second])
        assert combined == provenance.sha256_files([second, first])
        assert str(first) not in combined
        assert len(provenance.sha256_file(first)) == 64


def test_provenance_extracts_gpu_name_and_core_count() -> None:
    displays = {
        "SPDisplaysDataType": [
            {"sppci_model": "Apple Test GPU", "sppci_cores": "80"},
        ],
    }
    assert provenance.graphics_device(displays) == {
        "device": "Apple Test GPU",
        "gpu_core_count": 80,
    }


def test_runner_refuses_incomplete_implementation_identity() -> None:
    args = argparse.Namespace(implementation="forja", resolved_config=None, weights=[])
    try:
        serving_bench.validate_identity(args, {"tokenizer": Path("tokenizer.json")})
    except ValueError as error:
        assert "resolved-config" in str(error)
    else:
        raise AssertionError("incomplete Forja identity was accepted")


def test_compare_refuses_every_workload_identity_mismatch() -> None:
    compare = comparison_module()
    case = {"name": "case", "prompt_tokens": 7, "concurrency": 1, "sampling": "greedy"}
    record = {
        "schema_version": 1,
        "suite_sha256": "suite",
        "prompt_sha256": "prompt",
        "model": {"id": "model", "revision": "revision"},
        "sampling_seed": 0,
        "warmups": 3,
        "connection_policy": "reuse",
        "results": [{"case": case}],
    }
    for field, value in [
        ("schema_version", 2),
        ("suite_sha256", "other"),
        ("prompt_sha256", "other"),
        ("sampling_seed", 1),
        ("warmups", 4),
        ("connection_policy", "new"),
    ]:
        changed = {**record, field: value}
        try:
            compare.comparable(record, changed)
        except ValueError as error:
            assert "differs" in str(error)
        else:
            raise AssertionError(f"{field} mismatch was accepted")
    for field, value in [("id", "other"), ("revision", "other")]:
        changed = json.loads(json.dumps(record))
        changed["model"][field] = value
        try:
            compare.comparable(record, changed)
        except ValueError as error:
            assert "differs" in str(error)
        else:
            raise AssertionError(f"model {field} mismatch was accepted")
    for field, value in [("concurrency", 2), ("sampling", "temperature"), ("prompt_tokens", 8)]:
        changed = json.loads(json.dumps(record))
        changed["results"][0]["case"][field] = value
        try:
            compare.comparable(record, changed)
        except ValueError as error:
            expected = f"{field} differs" if field in {"concurrency", "sampling"} else "requests differ"
            assert str(error) == expected
        else:
            raise AssertionError(f"{field} mismatch was accepted")


def test_compare_reports_improvement_above_one_for_both_metric_directions() -> None:
    compare = comparison_module()
    case = {"name": "case", "prompt_tokens": 7, "concurrency": 1}
    latency = {"p50": 2.0, "p90": 2.0, "p99": 2.0}
    baseline = {"results": [{"case": case, "requests_per_second": 10.0, "prompt_tokens_per_second": 10.0, "output_tokens_per_second": 10.0, "ttft_seconds": latency, "itl_seconds": latency, "e2e_seconds": latency}]}
    faster = {"results": [{"case": case, "requests_per_second": 20.0, "prompt_tokens_per_second": 20.0, "output_tokens_per_second": 20.0, "ttft_seconds": {key: 1.0 for key in latency}, "itl_seconds": {key: 1.0 for key in latency}, "e2e_seconds": {key: 1.0 for key in latency}}]}
    reported = {metric: value for _case, metric, value in compare.rows(baseline, faster)}

    assert reported["requests_per_second"] == 2.0
    assert reported["ttft_seconds.p50"] == 2.0
