import copy

import conversation_session
import pytest


class Check:
    def prompt(self, messages, *_args):
        return {'prompt_tokens': len(messages) * 100}


class Client:
    def __init__(self): self.messages = []
    def chat(self, messages, _tokens, _options):
        self.messages.append(copy.deepcopy(messages))
        return {'error': None, 'prompt_tokens': len(messages) * 100, 'output_tokens': 7,
                'decode_window': [11., 13.], 'request_window': [10., 14.],
                'ttft_seconds': 1., 'e2e_seconds': 4., 'cached_tokens': 0 if len(self.messages) == 1 else 500,
                'assistant_message': {'role': 'assistant', 'content': 'actual reply'}}


def test_actual_replies_and_server_token_counts_define_the_record():
    client = Client()
    suite = {'tool_output_tokens': [7, 33], 'scripted_assistant_tokens': 1,
             'output_tokens': 7, 'max_context': 4097}
    raw = []
    rows = conversation_session.run(client, suite, 'sample-0', Check(), {}, raw.append)
    assert client.messages[1][4] == {'role': 'assistant', 'content': 'actual reply'}
    assert rows[1]['generated_reply_hit']
    assert rows[1]['reply_recompute_hit'] is None
    assert rows[0]['decode_tokens_per_second'] == 2
    reports = conversation_session.reports([rows, rows])
    assert reports[0]['cache_hits'] == 0
    assert reports[1]['cache_hits'] == 2
    assert len(raw) == 2


def test_recompute_wait_is_outside_http_timing(monkeypatch):
    observations = iter([{'pending': 1, 'completed': 4}, {'pending': 0, 'completed': 5}])
    monkeypatch.setattr(conversation_session, 'metrics', lambda _identity: next(observations))
    monkeypatch.setattr(conversation_session.time, 'sleep', lambda _seconds: None)
    observed, wait = conversation_session.await_idle({'cache_metrics': {'pending': 'pending', 'completed': 'completed'}}, {'completed': 4})
    assert observed['completed'] == 5 and wait >= 0


def test_prompt_identity_failure_stops_before_another_turn():
    class Wrong(Check):
        def prompt(self, *_args): return {'prompt_tokens': 1}
    with pytest.raises(ValueError, match='tokenizer'):
        conversation_session.run(Client(), {'tool_output_tokens': [1], 'scripted_assistant_tokens': 1,
            'output_tokens': 7, 'max_context': 4097}, 'sample', Wrong(), {}, lambda _row: None)


@pytest.mark.parametrize('turn, cached, message', [
    (1, 64, 'cold conversation unexpectedly hit'),
    (2, 0, 'cached conversation path did not run'),
])
def test_cache_path_refusals_happen_before_retaining_bad_turn(turn, cached, message):
    class WrongCache(Client):
        def chat(self, messages, tokens, options):
            result = super().chat(messages, tokens, options)
            if len(self.messages) == turn:
                result['cached_tokens'] = cached
            return result
    retained = []
    with pytest.raises(ValueError, match=message):
        conversation_session.run(WrongCache(), {'tool_output_tokens': [1, 7],
            'scripted_assistant_tokens': 1, 'output_tokens': 7, 'max_context': 4097},
            'sample', Check(), {}, retained.append)
    assert len(retained) == turn - 1
