#!/usr/bin/env python3
"""Frozen-workload native MLX benchmark with fresh per-block residency."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_stats import stats  # noqa: E402
from native_runtime import Runtime, engine_hash  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['bench'])
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--pp', type=int, required=True)
    parser.add_argument('--tg', type=int, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--raw-samples', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if min(args.pp, args.tg, args.reps) < 1 or args.context < 2 or args.warmups < 0:
        parser.error('invalid native workload')
    return args


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def request(line, expected):
    value = json.loads(line)
    if set(value) != {'id','output'} or value['id'] != expected:
        raise ValueError('worker requests must have ordered ids and an output path')
    output = Path(value['output'])
    if not output.is_absolute() or output.exists():
        raise ValueError('worker requires a new absolute output path')
    return value


def write(path, record):
    with Path(path).open('x') as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write('\n')


def emit(kind, **fields):
    print(f'FORJA_BENCH_{kind} '+json.dumps({'protocol':3, 'pid':os.getpid(),
        'runner_loads':1,'instance_loads':{'single':1}, **fields}), flush=True)


def summary(samples, tokens):
    return {'tokens_per_second': {'wall': stats([tokens/s['wall_seconds'] for s in samples]), 'gpu':None},
            'wall_time_seconds':stats([s['wall_seconds'] for s in samples]),'gpu_time_seconds':None}


def measure(runtime, args, identity, source):
    started = time.perf_counter()
    pp, tg, digest, phases = runtime.sample(args)
    elapsed = time.perf_counter() - started
    measured = sum(s['wall_seconds'] for s in pp+tg)
    phases.update(measured_seconds=measured, total_seconds=elapsed,
        request_setup_seconds=max(0,elapsed-measured-sum(phases.values())))
    record = {'schema_version':1,'implementation':'mlx_lm', 'kind':'native-model-benchmark',
        'model':identity['model'], 'identity':identity, 'provenance':source, 'inputs':[{'engine_sha256':source['engine_sha256'],
            'weights_sha256':identity['weights_sha256']}],
        'measurement_mode':{'sampling':'fixed','reps':args.reps,'execution':'resident-worker' if args.worker else 'process-per-leg'},
        'settings':{'pp':args.pp,'tg':args.tg,'context_start':args.context,'prefill_step_size':2048,
            'greedy':True,'seed':None,'cache_state':'fresh-request','cache_types':runtime.cache_types,
            'workload_max_context':max(args.pp+1,args.context+args.tg),
            'timed_decode_input_positions':[args.context,args.context+args.tg-1]},
        'results':[{'pp':summary(pp,args.pp),'tg':summary(tg,args.tg),
            'output_digest':digest,'output_digest_algorithm':'sha256-u32le-generated-token-ids',
            'raw_samples':{'pp':pp,'tg':tg}}]}
    if args.worker:
        record['worker']={'protocol':3,'pid':os.getpid(),'runner_loads':1,'instance_loads':{'single':1},
            'cache_state':'fresh-request','configured_priming_passes':args.warmups,
            'priming_policy':'once-per-worker-shape; at most two decode steps','cache_builds':runtime.cache_builds,
            'phase_timers':phases}
    return record


def main():
    args = arguments()
    identity = json.loads(args.identity.read_text())
    if args.model_dir.resolve() != Path(identity['model_directory']).resolve():
        raise ValueError('model directory differs from frozen identity')
    for item in identity['artifacts'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('frozen native artifact mismatch')
    site = Path(importlib.util.find_spec('mlx_lm').origin).parent.parent
    source = {'binary_sha256':sha(__file__), 'runtime_sha256':sha(Path(__file__).with_name('native_runtime.py')),
        'engine_sha256':engine_hash(site),'python_sha256':sha(sys.executable),'python':sys.version,
        'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm'),
        'os':platform.platform(), 'identity_sha256':sha(args.identity)}
    started = time.time()
    runtime = Runtime(args)
    source['device'] = runtime.mx.device_info()
    if not args.worker:
        write(args.json, measure(runtime,args,identity,source))
        return
    emit('READY', started_epoch=started, ready_epoch=time.time())
    for expected, line in enumerate(sys.stdin, 1):
        item = request(line, expected)
        record = measure(runtime,args,identity,source)
        write(item['output'],record)
        emit('RESULT',id=item['id'],cache_builds=runtime.cache_builds)


if __name__ == '__main__':
    main()
