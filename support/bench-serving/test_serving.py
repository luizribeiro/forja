"""Tests for the serving benchmark client."""

import tempfile
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import http_client
import suite


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
