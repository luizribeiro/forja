from copy import deepcopy

import pytest

from gate_records import verify_fresh_worker
from gate_worker import measurement_protocol, worker_policy


def test_fresh_worker_protocol_binds_residency_and_bounded_priming():
    adapter = {"command": ["/frozen/mlx-native.py", "bench"], "worker_protocol": 3,
               "cache_state": "fresh-request", "expected_provenance": {
                   "/provenance/binary_sha256": "binary", "/inputs/0/engine_sha256": "engine"}}
    row = {"baseline": adapter, "candidate": deepcopy(adapter)}
    assert worker_policy(row) is None
    protocol = measurement_protocol(row, True, 1)
    assert protocol['execution'] == 'resident-block-workers'
    assert protocol['worker_protocol'] == 3
    assert protocol['priming'] == 'once-per-worker-shape; two decode steps'
    row['candidate']['cache_state'] = 'prefix-fork-per-worker'
    with pytest.raises(ValueError, match='cache-state policy'):
        worker_policy(row)


def test_fresh_worker_proof_rejects_missing_or_nonfinite_timers():
    worker = {'cache_state': 'fresh-request', 'configured_priming_passes': 1,
              'priming_policy': 'once-per-worker-shape; at most two decode steps',
              'phase_timers': {'priming_seconds': 1., 'measured_seconds': 2.}}
    verify_fresh_worker(worker, 1)
    for key, value in [('cache_state', 'cold-process'), ('configured_priming_passes', 3),
                       ('phase_timers', {}), ('phase_timers', {'seconds': float('nan')})]:
        with pytest.raises(ValueError, match='proof'):
            verify_fresh_worker({**worker, key: value}, 1)
