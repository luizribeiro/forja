"""Packaged HTTP benchmark entry point used by the native CLI."""
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import conversation_identity
import conversation_session
import http_client
import multi_turn
import provenance
import serving_bench


def write(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def run(options):
    output = Path(options['json'])
    if output.exists():
        raise ValueError('refusing to overwrite a benchmark record')
    identity = conversation_identity.load(options['identity'])
    suite, suite_hash = multi_turn.load_suite(Path(options['suite']))
    check = conversation_identity.TokenCheck(identity)
    command = json.loads(Path(options['server']).read_text())
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError('server command must be a nonempty JSON argv array')
    if identity['model']['id'] != suite['model'] or identity['model']['revision'] != suite['revision']:
        raise ValueError('suite and frozen server model differ')
    server = serving_bench.Server(command, options['url'], output.with_suffix('.server.log'))
    client = http_client.OpenAIClient(options['url'], suite['model'])
    raw = output.with_suffix('.raw.jsonl')
    runs, processes = [], []
    sampled_rss = 0
    started = time.time()
    try:
        with raw.open('x') as stream:
            for repetition in range(options['reps']):
                server.start()
                client.reset_connection()
                process = {'pid': server.process.pid, 'ready_epoch': time.time()}
                prime_started = time.time()
                for _ in range(min(1, options['warmups'])):
                    sample = client.chat([{'role': 'user', 'content': 'Benchmark initialization.'}],
                                         2, identity.get('chat_template_kwargs', {}))
                    if sample['error']:
                        raise RuntimeError(sample['error'])
                process['priming_seconds'] = time.time() - prime_started
                def retain(row):
                    stream.write(json.dumps({'session': repetition, 'turn': row}, allow_nan=False) + '\n')
                    stream.flush()
                runs.append(conversation_session.run(client, suite, 'sample-0', check, identity, retain))
                sampled_rss = max(sampled_rss, server.rss_bytes() or 0)
                process['finished_epoch'] = time.time()
                processes.append(process)
                server.stop()
    finally:
        server.stop()
    versions = {}
    for package in ['transformers', 'tokenizers', 'mlx', 'mlx-lm']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'schema_version': 1, 'kind': 'http-conversation-benchmark',
        'implementation': identity['implementation'], 'server_identity': identity,
        'model': identity['model'], 'suite_sha256': suite_hash,
        'server_command': command, 'server_command_sha256': conversation_identity.sha256(options['server']),
        'provenance': {**options['provenance'], 'python': sys.version,
                       'python_sha256': conversation_identity.sha256(sys.executable),
                       'packages': versions, **provenance.machine(), 'os_build': platform.version()},
        'measurement_mode': {'sampling': 'fixed', 'execution': 'process-per-leg',
            'cache_state': 'fresh-server-per-conversation', 'unit': 'whole-conversation',
            'reps': options['reps'], 'warmups': min(1, options['warmups']),
            'priming': 'one two-token request per fresh server'},
        'processes': processes, 'started_epoch': started, 'finished_epoch': time.time(),
        'server_startup_seconds': server.startup_seconds, 'sampled_peak_rss_bytes': sampled_rss,
        'sampled_rss_is_high_water_mark': False, 'gpu_time_seconds': None,
        'prompt_cache_reused': all(row['cache_hit'] for run in runs for row in run[1:]),
        'notes': ['Each repetition is a fresh whole conversation; turns are not independent trials.',
            'Actual assistant messages are appended to history. CPU rendering verifies tokens before HTTP.',
            'TTFT/ITL refer to visible SSE content events. Decode rate uses server completion_tokens minus one over first-to-last content event; token grouping is not inferred.',
            'Idle recompute waits and startup are outside TTFT/decode windows and retained in whole-session wall time.',
            'A null reply_recompute_hits field means the server has no corresponding recompute metric.'],
        'turns': conversation_session.reports(runs)}


def main():
    options = json.loads(Path(sys.argv[1]).read_text())
    try:
        record = run(options)
        write(options['json'], record)
    except BaseException as error:
        write(Path(options['json']).with_suffix('.failure.json'),
              {'error': type(error).__name__, 'message': str(error), 'options': options})
        raise


if __name__ == '__main__':
    main()
