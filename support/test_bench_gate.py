"""Exercise command persistence and provenance checks using synthetic native records."""

from copy import deepcopy
import json
import sys

import pytest

from gate_records import command_sampler


def test_native_adapter_preserves_records_and_checks_hashes(tmp_path):
    script = tmp_path / "synthetic.py"
    script.write_text('''import json,sys,time
from pathlib import Path
now=time.time()
Path(sys.argv[1]).write_text(json.dumps({"implementation":"forja","provenance":{"binary_sha256":"host"},
"inputs":[{"engine_sha256":"guest"}],"config":"[bench]\\nreps="+sys.argv[2],
"results":[{"output_digest":"exact","score":100,"window":[now,now+.0001]}]}))
''')
    adapter = {"command": [sys.executable, str(script), "{output}", "{reps}", "bench.warmups={warmups}"],
        "metric": "/results/0/score", "window": "/results/0/window",
        "expected_provenance": {"/provenance/binary_sha256": "host", "/inputs/0/engine_sha256": "guest"}}
    row = {"id": "row", "baseline": adapter, "candidate": deepcopy(adapter)}
    sample = command_sampler(row, tmp_path, [], 1, True)
    packet = sample("A", 1, "primary", 0, 0)
    assert packet["score"] == 100
    assert packet["output_digests"] == ["exact"]
    event = json.loads((tmp_path / "primary-00-0-A.command.json").read_text())
    assert event["command"][-1] == "bench.warmups=1"
    assert json.loads((tmp_path / "primary-00-0-A.json").read_text())["config"] == "[bench]\nreps=1"
    with pytest.raises(ValueError, match="repeat"):
        sample("A", 1, "primary", 0, 0)
    row["candidate"]["expected_provenance"]["/inputs/0/engine_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        sample("B", 1, "primary", 0, 1)
