from pathlib import Path
from types import SimpleNamespace

import http_entry
import pytest


def test_http_runner_preserves_sessions_and_cleans_up_on_failure(monkeypatch, tmp_path):
    suite = Path(__file__).resolve().parents[2] / 'bench/suites/multi-turn.toml'
    command = tmp_path / 'command.json'
    command.write_text('["fake-server"]')
    identity = {'model': {'id': 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
                         'revision': '6e302ea604ad9ab206367e2c501d1571023e7b6d'},
                'implementation': 'test'}
    monkeypatch.setattr(http_entry.conversation_identity, 'load', lambda _path: identity)
    monkeypatch.setattr(http_entry.conversation_identity, 'TokenCheck', lambda _identity: None)
    monkeypatch.setattr(http_entry.provenance, 'machine', lambda: {})
    events = []
    class Server:
        startup_seconds = 0
        def __init__(self, *_args): self.process = SimpleNamespace(pid=7)
        def start(self): events.append('start')
        def stop(self): events.append('stop')
        def rss_bytes(self): return 33
    monkeypatch.setattr(http_entry.serving_bench, 'Server', Server)
    def fail(*_args): raise ValueError('failed turn')
    monkeypatch.setattr(http_entry.conversation_session, 'run', fail)
    with pytest.raises(ValueError, match='failed turn'):
        http_entry.run({'json': str(tmp_path/'record.json'), 'identity': 'identity',
            'suite': str(suite), 'server': str(command), 'url': 'http://127.0.0.1:1',
            'reps': 1, 'warmups': 0, 'provenance': {}})
    assert events == ['start', 'stop']


def test_successful_conversations_record_provenance_and_turns(monkeypatch, tmp_path):
    suite = Path(__file__).resolve().parents[2] / 'bench/suites/multi-turn.toml'
    command = tmp_path / 'command.json'
    command.write_text('["fake-server"]')
    identity = {'model': {'id': 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
                          'revision': '6e302ea604ad9ab206367e2c501d1571023e7b6d'},
                'implementation': 'test'}
    monkeypatch.setattr(http_entry.conversation_identity, 'load', lambda _path: identity)
    monkeypatch.setattr(http_entry.conversation_identity, 'TokenCheck', lambda _identity: None)
    monkeypatch.setattr(http_entry.provenance, 'machine', lambda: {'machine': 'test-machine'})
    calls = []
    class Server:
        startup_seconds = 1
        def __init__(self, *_args): self.process = SimpleNamespace(pid=7)
        def start(self): calls.append('start')
        def stop(self): calls.append('stop')
        def rss_bytes(self): return 33
    class Client:
        def __init__(self, *_args): pass
        def reset_connection(self): pass
        def chat(self, _messages, budget, _kwargs):
            assert budget == 2
            calls.append('prime')
            return {'error': None}
    monkeypatch.setattr(http_entry.serving_bench, 'Server', Server)
    monkeypatch.setattr(http_entry.http_client, 'OpenAIClient', Client)
    monkeypatch.setattr(http_entry.conversation_session, 'run',
                        lambda *_args: [{'cache_hit': False}, {'cache_hit': True}])
    monkeypatch.setattr(http_entry.conversation_session, 'reports', lambda runs: [{'runs': len(runs)}])
    record = http_entry.run({'json': str(tmp_path/'record.json'), 'identity': 'identity',
        'suite': str(suite), 'server': str(command), 'url': 'http://127.0.0.1:1',
        'reps': 2, 'warmups': 1, 'provenance': {'commit': 'frozen'}})
    assert record['schema_version'] == 1 and record['kind'] == 'http-conversation-benchmark'
    assert record['measurement_mode']['unit'] == 'whole-conversation'
    assert record['measurement_mode']['reps'] == 2 and record['measurement_mode']['warmups'] == 1
    assert record['turns'] == [{'runs': 2}] and record['prompt_cache_reused']
    assert record['provenance']['commit'] == 'frozen'
    assert record['provenance']['machine'] == 'test-machine'
    assert record['server_command'] == ['fake-server'] and len(record['server_command_sha256']) == 64
    assert len(record['processes']) == 2 and record['sampled_peak_rss_bytes'] == 33
    assert len(record['notes']) == 5 and 'Actual assistant messages' in record['notes'][1]
    assert calls == ['start', 'prime', 'stop', 'start', 'prime', 'stop', 'stop']


@pytest.mark.parametrize('field', ['id', 'revision'])
def test_suite_model_mismatch_stops_before_server_launch(monkeypatch, tmp_path, field):
    suite = Path(__file__).resolve().parents[2] / 'bench/suites/multi-turn.toml'
    command = tmp_path / 'command.json'
    command.write_text('["fake-server"]')
    identity = {'model': {'id': 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
                          'revision': '6e302ea604ad9ab206367e2c501d1571023e7b6d'},
                'implementation': 'test'}
    identity['model'][field] = 'wrong'
    monkeypatch.setattr(http_entry.conversation_identity, 'load', lambda _path: identity)
    monkeypatch.setattr(http_entry.conversation_identity, 'TokenCheck', lambda _identity: None)
    monkeypatch.setattr(http_entry.serving_bench, 'Server', lambda *_args: pytest.fail('server launched'))
    with pytest.raises(ValueError, match='suite and frozen server model differ'):
        http_entry.run({'json': str(tmp_path/'record.json'), 'identity': 'identity',
            'suite': str(suite), 'server': str(command), 'url': 'http://127.0.0.1:1',
            'reps': 1, 'warmups': 0, 'provenance': {}})
