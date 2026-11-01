import sys
import time

import serving_bench


def test_owned_server_keeps_launch_output_across_restarts(monkeypatch, tmp_path):
    monkeypatch.setattr(serving_bench, "ready", lambda _url: True)
    log = tmp_path / "server.log"
    server = serving_bench.Server(
        [sys.executable, "-u", "-c", "print('loaded'); import time; time.sleep(30)"],
        "http://127.0.0.1:1", log)
    try:
        for count in [1, 2]:
            server.start()
            deadline = time.monotonic() + 5
            while log.read_text().count("loaded") < count and time.monotonic() < deadline:
                time.sleep(.01)
            assert log.read_text().count("loaded") == count
    finally:
        server.stop()
