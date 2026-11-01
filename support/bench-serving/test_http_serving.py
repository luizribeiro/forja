from pathlib import Path

import http_serving
import pytest


def test_serving_cli_keeps_identity_and_omits_greedy_seed(monkeypatch, tmp_path):
    suite = tmp_path / 'suite.toml'
    command = tmp_path / 'server.json'
    command.write_text('["server", "argument with spaces"]')
    monkeypatch.setattr(http_serving.suite, 'load', lambda _: (
        {'model': 'model', 'revision': 'revision', 'seed': 0}, [{'sampling': 'greedy'}], 'hash'))
    seen = []
    def run(args):
        seen.append(args)
        return {'provenance': {'server': 'identity'}}
    monkeypatch.setattr(http_serving.serving_bench, 'run', run)
    identity = {'model': {'id': 'model', 'revision': 'revision'}, 'implementation': 'mlx_lm',
        'repository': str(tmp_path), 'artifacts': {}, 'weights': ['weights']}
    options = {'suite': str(suite), 'server': str(command), 'json': str(tmp_path/'result.json'),
        'url': 'http://localhost:1', 'reps': 1, 'warmups': 1, 'raw_samples': True,
        'provenance': {'binary_sha256': 'cli'}}
    record = http_serving.run(options, identity)
    assert record['provenance']['binary_sha256'] == 'cli'
    assert record['server_identity'] == identity
    assert record['sampling_seed'] is None
    assert seen[0].server_command == "server 'argument with spaces'"
    assert seen[0].weights == [Path('weights')]
    assert seen[0].server_log == tmp_path/'result.server.log'
    identity['model']['revision'] = 'changed'
    with pytest.raises(ValueError, match='differ'):
        http_serving.run(options, identity)
