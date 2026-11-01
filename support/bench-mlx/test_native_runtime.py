import argparse

import native_runtime


def fake_runtime():
    runtime = object.__new__(native_runtime.Runtime)
    runtime.primed = False
    calls = []
    runtime.prefill = lambda count: calls.append(('pp', count)) or {'wall_seconds': 1}
    runtime.decode = lambda context, steps: (calls.append(('tg', context, steps)) or {'wall_seconds': 1}, [7])
    runtime.stream = lambda count, steps: [argparse.Namespace(token=7)] * steps
    runtime.mx = argparse.Namespace(synchronize=lambda: None)
    return runtime, calls


def test_tokens_match_the_common_workload_and_priming_is_bounded():
    assert native_runtime.synthetic_tokens(3, 151936) == [131128, 117465, 70310]
    runtime, calls = fake_runtime()
    args = argparse.Namespace(pp=512, context=4096, tg=128, reps=1, warmups=1, worker=True)
    first = runtime.sample(args)
    runtime.sample(args)
    assert calls == [('pp',512), ('tg',4096,2), ('pp',512), ('tg',4096,128), ('pp',512), ('tg',4096,128)]
    assert len(first[2]) == 64
    assert len(first[0]) == len(first[1]) == 1


def test_classic_priming_retains_every_configured_full_pass():
    runtime, calls = fake_runtime()
    args = argparse.Namespace(pp=512, context=4096, tg=128, reps=1, warmups=3, worker=False)
    runtime.sample(args)
    runtime.sample(args)
    assert calls == [('pp',512), ('tg',4096,128)] * 8


def test_decode_times_the_same_input_positions_with_generator_lookahead():
    runtime = object.__new__(native_runtime.Runtime)
    submitted, boundaries, requested = [], [], []
    def stream(count, steps):
        requested.append((count, steps))
        for index in range(steps):
            submitted.append(count + index)
            yield argparse.Namespace(token=index)
    runtime.stream = stream
    runtime.mx = argparse.Namespace(synchronize=lambda: boundaries.append(list(submitted)))
    sample, tokens = runtime.decode(4096, 7)
    assert requested == [(4095, 8)]
    assert boundaries[0] == [4095]
    assert boundaries[1][len(boundaries[0]):] == list(range(4096, 4103))
    assert tokens == list(range(1, 8))
    assert sample['wall_seconds'] > 0


def test_prefill_window_excludes_the_unmeasured_lookahead_drain(monkeypatch):
    runtime = object.__new__(native_runtime.Runtime)
    epoch = [100.0]
    def stream(_count, _steps):
        try:
            yield argparse.Namespace(prompt_tps=64)
        finally:
            epoch[0] += 20
    def drain(): epoch[0] += 10
    runtime.stream = stream
    runtime.mx = argparse.Namespace(synchronize=drain)
    monkeypatch.setattr(native_runtime.time, 'time', lambda: epoch[0])
    sample = runtime.prefill(512)
    assert sample['measurement_window'] == [92, 100]
    assert sample['wall_seconds'] == 8
    assert epoch[0] == 130
