"""Exercise the bounded HTTP protocol through the complete benchmark runner."""
import priming


def test_bounded_http_keeps_measured_cases_and_restarts_cold_server(monkeypatch, tmp_path):
    import argparse
    import time
    import serving_bench

    cold = {"name": "cold", "route": "completions", "prompt_tokens": 512,
            "output_tokens": 128, "concurrency": 1, "sampling": "greedy", "cold": True}
    warm = cold | {"name": "warm", "route": "chat", "cold": False, "warm_prefix_tokens": 2048}
    settings = {"model": "model", "revision": "rev", "seed": 0,
                "connection_policy": "reuse", "warmups": 3, "repetitions": 1}
    calls, starts = [], []

    class Server:
        process = None
        startup_seconds = 0.0

        def __init__(self, *_args):
            pass

        def start(self):
            starts.append(len(calls))
            self.process = object()

        def stop(self):
            self.process = None

        def rss_bytes(self):
            return 0

    class Client:
        def __init__(self, *_args):
            pass

        def reset_connection(self):
            pass

        def request(self, case):
            calls.append(dict(case))
            now = time.perf_counter()
            return priming.http_client.sample(now, now, now + 1e-6, [], 512, case["output_tokens"],
                                              64 if case.get("warm_prefix_tokens") else 0, False, "same")

    monkeypatch.setattr(serving_bench.suite, "load", lambda _path: (settings, [cold, warm], "suite"))
    monkeypatch.setattr(serving_bench.suite, "expand", lambda cases: cases)
    monkeypatch.setattr(serving_bench, "Server", Server)
    monkeypatch.setattr(serving_bench.http_client, "OpenAIClient", Client)
    monkeypatch.setattr(serving_bench, "scrape_metrics", lambda _url: {"forja_engine_gpu_time_nanoseconds": 1.0})
    monkeypatch.setattr(serving_bench, "validate_identity", lambda *_args: None)
    monkeypatch.setattr(serving_bench.provenance, "collect", lambda *_args: {})
    args = argparse.Namespace(suite=tmp_path, artifact=[], server_command="server", url="http://local",
                              priming="bounded", implementation="forja", repository=tmp_path,
                              forja_repository=tmp_path, weights=[], resolved_config=None, metrics_url=None)
    record = serving_bench.run(args)
    assert [case["output_tokens"] for case in calls] == [2, 128, 1, 2, 128]
    assert [result["case"] for result in record["results"]] == [cold, warm]
    assert starts == [0, 1]
    assert record["results"][0]["server_metrics"]["gpu_seconds"] is None
    assert record["results"][1]["server_metrics"]["gpu_seconds"] == 0.0
    assert record["priming"]["paths"][0]["cached_tokens"] == 0
    assert record["priming"]["paths"][0]["fresh_server_before_measurement"]
    assert record["phase_timers"]["measured_seconds"] > 0
