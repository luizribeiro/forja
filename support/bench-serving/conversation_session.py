"""Actual-reply conversations with explicit cache and recompute observations."""
import copy
import time

import multi_turn
import serving_bench
from bench_stats import stats


def metrics(identity):
    config = identity.get('cache_metrics', {})
    return serving_bench.scrape_metrics(config.get('url')) or {}


def await_idle(identity, before):
    config = identity.get('cache_metrics', {})
    started = time.perf_counter()
    while True:
        after = metrics(identity)
        if not config or (after[config['pending']] == 0 and
                          after[config['completed']] > before[config['completed']]):
            return after, time.perf_counter() - started
        if time.perf_counter() - started > 120:
            raise TimeoutError('reply recompute did not reach its recorded idle boundary')
        time.sleep(.02)


def run(client, suite, label, check, identity, retain):
    messages = [{'role': 'system', 'content': f'Deterministic benchmark {label}.'}]
    samples = []
    for turn, tool_tokens in enumerate(suite['tool_output_tokens'], 1):
        messages.extend(multi_turn.turn_messages(label, turn, tool_tokens, suite['scripted_assistant_tokens']))
        proof = check.prompt(messages, suite['output_tokens'], suite['max_context'])
        before = metrics(identity)
        epoch = time.time()
        result = client.chat(messages, suite['output_tokens'], identity.get('chat_template_kwargs', {}))
        if result['error'] is not None:
            raise RuntimeError(result['error'])
        if result['prompt_tokens'] != proof['prompt_tokens']:
            raise ValueError('server and original tokenizer disagree on prompt identity')
        first, last = result['decode_window']
        if result['output_tokens'] < 2 or last <= first:
            raise ValueError('decode measurement requires two content events and server token counts')
        decode_seconds = result['e2e_seconds'] - result['ttft_seconds']
        if decode_seconds <= 0:
            raise ValueError('nonpositive HTTP decode interval')
        after, idle = await_idle(identity, before)
        config = identity.get('cache_metrics', {})
        previous = samples[-1] if samples else None
        reply_hit = bool(previous and result['cached_tokens'] > previous['prompt_tokens'])
        result.update(turn=turn, prompt_proof=proof, request_messages=copy.deepcopy(messages),
            decode_tokens_per_second=(result['output_tokens'] - 1) / decode_seconds,
            cache_hit=result['cached_tokens'] > 0, generated_reply_hit=reply_hit,
            reply_recompute_hit=reply_hit if config else None,
            recompute_completed=after[config['completed']] - before[config['completed']] if config else None,
            metrics_before=before, metrics_after=after, idle_wait_seconds=idle,
            windows={'ttft': [epoch, epoch + result['ttft_seconds']],
                     'decode': [epoch + result['ttft_seconds'], epoch + result['e2e_seconds']]})
        if turn == 1 and result['cache_hit']:
            raise ValueError('cold conversation unexpectedly hit the prefix cache')
        if previous and not result['cache_hit']:
            raise ValueError('cached conversation path did not run')
        samples.append(result)
        retain(result)
        messages.append(result['assistant_message'])
    return samples


def reports(runs):
    fields = ['ttft_seconds', 'decode_tokens_per_second', 'prompt_tokens', 'output_tokens', 'cached_tokens']
    return [{'turn': index + 1, 'cache_phase': 'cold' if index == 0 else 'cached',
             **{field: stats([run[index][field] for run in runs]) for field in fields},
             'cache_hits': sum(run[index]['cache_hit'] for run in runs),
             'generated_reply_hits': sum(run[index]['generated_reply_hit'] for run in runs),
             'reply_recompute_hits': sum(run[index]['reply_recompute_hit'] for run in runs)
                 if all(run[index]['reply_recompute_hit'] is not None for run in runs) else None,
             'samples': [run[index] for run in runs]} for index in range(len(runs[0]))]
