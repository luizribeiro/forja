#!/usr/bin/env python3
"""Observe native batching and enforce the configured prompt-cache byte limit."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def install(server, log, ceiling):
    base_cache, base_batch, base_provider = server.LRUPromptCache, server.BatchGenerator, server.ModelProvider
    class ConfiguredCache(base_cache):
        def __init__(self, max_size=10):
            super().__init__(max_size=max_size, max_bytes=ceiling)
    if log is None:
        server.LRUPromptCache = ConfiguredCache
        return
    def emit(event, **fields):
        with Path(log).open('a') as output:
            output.write(json.dumps({'event':event,'pid':os.getpid(),'epoch':time.time(),**fields})+'\n')
    class Cache(ConfiguredCache):
        def fetch_nearest_cache(self, model, tokens):
            cache, rest = super().fetch_nearest_cache(model,tokens)
            emit('prefix_fetch',prompt_tokens=len(tokens),cached_tokens=len(tokens)-len(rest),
                 cache_types=[type(c).__name__ for c in cache or []],
                 trimmable=all(c.is_trimmable() for c in cache) if cache else None,
                 retained_bytes=self.nbytes)
            return cache, rest
        def insert_cache(self, model, tokens, prompt_cache, **options):
            result = super().insert_cache(model,tokens,prompt_cache,**options)
            emit('prefix_insert',tokens=len(tokens),cache_type=options.get('cache_type','assistant'),
                 cache_types=[type(c).__name__ for c in prompt_cache],
                 state_dtypes=sorted({str(v.dtype) for c in prompt_cache for v in c.state if hasattr(v,'dtype')}),
                 retained_bytes=self.nbytes,byte_ceiling=self.max_bytes,entries=len(self))
            return result
    class Batch(base_batch):
        def next(self):
            result = super().next()
            emit('batch_step',prompt_ids=sorted({str(r.uid) for r in result[0]}),
                 decode_ids=sorted({str(r.uid) for r in result[1]}))
            return result
    class Provider(base_provider):
        def __init__(self, args):
            emit('effective_config',settings=vars(args),prefix_cache_constructor_bytes=ceiling,
                 versions={name:importlib.metadata.version(name) for name in ['mlx','mlx-lm']})
            super().__init__(args)
    server.LRUPromptCache, server.BatchGenerator, server.ModelProvider = Cache, Batch, Provider


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--audit-log',type=Path)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--prompt-cache-bytes',type=int,required=True)
    args, remaining = parser.parse_known_args()
    if args.prompt_cache_bytes < 0:
        raise ValueError('negative cache ceiling')
    log = args.audit_log.resolve() if args.audit_log else None
    os.chdir(args.workdir)
    from mlx_lm import server
    install(server,log,args.prompt_cache_bytes)
    sys.argv = ['mlx_lm.server',*remaining,'--prompt-cache-bytes',str(args.prompt_cache_bytes)]
    server.main()


if __name__ == '__main__':
    main()
