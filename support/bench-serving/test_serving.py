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

    def do_GET(self) -> None:
        body = b"forja_engine_gpu_time_nanoseconds 3000000000\nforja_pool_bytes 17\n"
        self.send_response(200)
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
        assert len(result["output_sha256"]) == 1
        streamed = client.request({**case, "stream": True})
        assert streamed["output_tokens"] == 2
        assert len(streamed["itl_seconds"]) == 1
        cancelled = client.request({**case, "stream": True, "cancel_after_tokens": 1})
        assert cancelled["cancelled"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cold_case_runs_setup_before_every_request() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    setups = []
    try:
        client = http_client.OpenAIClient(f"http://127.0.0.1:{server.server_port}", "model")
        case = {"name": "cold", "route": "completions", "prompt_tokens": 7, "output_tokens": 2, "concurrency": 1, "sampling": "greedy"}
        result = http_client.run_case(client, case, 2, 3, lambda: setups.append(None))
        assert result["requests"] == 3
        assert len(setups) == 5
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_runner_scrapes_resource_gauges_and_counter_deltas() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/metrics"
        samples = serving_bench.scrape_metrics(url)
        report = serving_bench.metric_report(
            {**samples, "forja_engine_gpu_time_nanoseconds": 1_000_000_000}, samples
        )
        assert report["gpu_seconds"] == 2.0
        assert report["memory"]["pool_bytes"] == 17
        assert report["memory"]["table_bytes"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_runner_initializes_client_when_first_case_is_cold(monkeypatch, tmp_path) -> None:
    case = {"name": "cold", "route": "completions", "prompt_tokens": 7, "output_tokens": 2, "concurrency": 1, "sampling": "greedy", "cold": True}
    settings = {"model": "model", "revision": "revision", "seed": 0, "connection_policy": "reuse", "warmups": 0, "repetitions": 1}

    class ManagedServer:
        last = None

        def __init__(self, _command, _url):
            self.process = None
            self.starts = 0
            ManagedServer.last = self

        def start(self):
            self.process = object()
            self.starts += 1

        def stop(self):
            self.process = None

        def rss_bytes(self):
            return 1

    class Client:
        def __init__(self, _url, _model):
            self.resets = 0

        def reset_connection(self):
            self.resets += 1

    client = Client("", "")
    monkeypatch.setattr(serving_bench, "Server", ManagedServer)
    monkeypatch.setattr(serving_bench.suite, "load", lambda _path: (settings, [case], "suite"))
    monkeypatch.setattr(serving_bench.suite, "expand", lambda cases: cases)
    monkeypatch.setattr(serving_bench.http_client, "OpenAIClient", lambda _url, _model: client)
    monkeypatch.setattr(
        serving_bench.http_client,
        "run_case",
        lambda _client, _case, _warmups, _repetitions, setup, begin: (
            setup(),
            begin(),
            {"case": case},
        )[2],
    )
    monkeypatch.setattr(serving_bench.provenance, "collect", lambda *_args: {})
    args = argparse.Namespace(
        implementation="mlx_lm",
        url="http://127.0.0.1:1",
        suite=tmp_path / "suite.toml",
        artifact=[("tokenizer", tmp_path / "tokenizer.json")],
        weights=[tmp_path / "weights.safetensors"],
        server_command="mlx_lm.server",
        repository=tmp_path,
        forja_repository=tmp_path,
        resolved_config=None,
        metrics_url=None,
    )

    serving_bench.run(args)

    assert ManagedServer.last.starts == 1
    assert client.resets == 1


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
        "schema_version": 2,
        "suite_sha256": "suite",
        "prompt_sha256": "prompt",
        "model": {"id": "model", "revision": "revision"},
        "sampling_seed": 0,
        "warmups": 3,
        "connection_policy": "reuse",
        "results": [{"case": case}],
    }
    for field, value in [
        ("schema_version", 1),
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
