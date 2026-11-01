import json
from types import SimpleNamespace

import server_adapter


def test_observer_preserves_native_steps_and_sets_only_requested_cache_bound(tmp_path):
    class Cache:
        def __init__(self,max_size,max_bytes): self.max_size,self.max_bytes,self.nbytes=max_size,max_bytes,0
        def fetch_nearest_cache(self,_model,tokens): return None,tokens
        def insert_cache(self,*_args,**_kwargs): return 'inserted'
        def __len__(self): return 1
    response = ([SimpleNamespace(uid=1),SimpleNamespace(uid=2)],[SimpleNamespace(uid=1)])
    class Batch:
        def next(self): return response
    module = SimpleNamespace(LRUPromptCache=Cache,BatchGenerator=Batch,ModelProvider=object)
    log = tmp_path/'audit.jsonl'
    server_adapter.install(module,log,4097)
    cache = module.LRUPromptCache(10)
    assert (cache.max_size,cache.max_bytes) == (10,4097)
    assert cache.fetch_nearest_cache('model',[1,7,33]) == (None,[1,7,33])
    assert cache.insert_cache('model',[1],[]) == 'inserted'
    assert module.BatchGenerator().next() is response
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]['cached_tokens'] == 0
    assert rows[1]['byte_ceiling'] == 4097
    assert rows[2]['prompt_ids'] == ['1','2']
    unobserved = SimpleNamespace(LRUPromptCache=Cache,BatchGenerator=Batch,ModelProvider=object)
    server_adapter.install(unobserved,None,4097)
    assert unobserved.BatchGenerator is Batch
    assert unobserved.ModelProvider is object
    assert unobserved.LRUPromptCache(10).max_bytes == 4097
