import hashlib
import json

import conversation_identity
import pytest


def test_identity_refuses_changed_artifacts(tmp_path):
    artifact = tmp_path / 'data'
    artifact.write_bytes(b'original')
    bound = {'path': str(artifact), 'sha256': hashlib.sha256(b'original').hexdigest()}
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps({'model': {'id': 'test'}, 'tokenizer_path': str(tmp_path),
        'effective_config': {}, 'artifacts': {k: bound for k in ['tokenizer','tokenizer_config','weights','server']}}))
    assert conversation_identity.load(path)['model']['id'] == 'test'
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch'):
        conversation_identity.load(path)


def test_context_limit_is_checked_before_http_and_preserves_tool_input():
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert messages[0]['tool_calls'][0]['function']['arguments'] == {'key': 7}
            return 'rendered'
        def encode(self, *_args, **_kwargs): return [1, 7, 33]
    check = object.__new__(conversation_identity.TokenCheck)
    check.tokenizer, check.options = Tokenizer(), {}
    messages = [{'tool_calls': [{'function': {'arguments': '{"key":7}'}}]}]
    assert check.prompt(messages, 4, 7)['maximum_context'] == 7
    assert isinstance(messages[0]['tool_calls'][0]['function']['arguments'], str)
    with pytest.raises(ValueError, match='ceiling'):
        check.prompt(messages, 4, 6)
