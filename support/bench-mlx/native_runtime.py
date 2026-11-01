"""One loaded MLX model with fresh request caches and bounded priming."""
import hashlib
import struct
import time
from pathlib import Path


def synthetic_tokens(count, vocab):
    state = 0x4D595DF4D0F33173
    result = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        result.append(state % vocab)
    return result


class Runtime:
    def __init__(self, args):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        self.mx, self.generate, self.make_cache = mx, stream_generate, make_prompt_cache
        self.model, self.tokenizer, config = load(args.model_dir, return_config=True)
        text = config.get('text_config', config)
        if max(args.pp + 1, args.context + args.tg) > text['max_position_embeddings']:
            raise ValueError('native workload exceeds model context')
        self.vocab = text['vocab_size']
        self.tokenizer._eos_token_ids = set()
        self.sampler = make_sampler(temp=0.0, top_k=0, top_p=1.0)
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.cache_builds = 0
        self.primed = False
        self.cache_types = {}

    def stream(self, count, steps):
        cache = self.make_cache(self.model)
        self.cache_builds += 1
        self.cache_types = {name: sum(type(value).__name__ == name for value in cache)
                            for name in sorted({type(value).__name__ for value in cache})}
        return iter(self.generate(self.model, self.tokenizer,
            synthetic_tokens(count, self.vocab), max_tokens=steps,
            sampler=self.sampler, prompt_cache=cache, prefill_step_size=2048))

    def prefill(self, count):
        responses = self.stream(count, 1)
        response = next(responses)
        ended = time.time()
        seconds = count / response.prompt_tps
        self.mx.synchronize()
        return {'wall_seconds': seconds, 'gpu_seconds': None,
                'measurement_window': [ended - seconds, ended]}

    def decode(self, context, steps):
        responses = self.stream(context - 1, steps + 1)
        next(responses)
        self.mx.synchronize()
        epoch, started = time.time(), time.perf_counter()
        tokens = [next(responses).token for _ in range(steps)]
        self.mx.synchronize()
        seconds = time.perf_counter() - started
        return {'wall_seconds': seconds, 'gpu_seconds': None,
                'measurement_window': [epoch, epoch + seconds]}, tokens

    def sample(self, args):
        phases = {'priming_seconds': 0.0}
        if args.warmups and (not args.worker or not self.primed):
            started = time.perf_counter()
            for _ in range(1 if args.worker else args.warmups):
                self.prefill(args.pp)
                self.decode(args.context, min(2, args.tg) if args.worker else args.tg)
            phases['priming_seconds'] = time.perf_counter() - started
            self.primed = True
        pp, tg = [], []
        for _ in range(args.reps):
            pp.append(self.prefill(args.pp))
            sample, _tokens = self.decode(args.context, args.tg)
            tg.append(sample)
        started = time.perf_counter()
        probe = [response.token for response in self.stream(args.context - 1, args.tg)]
        self.mx.synchronize()
        phases['output_probe_seconds'] = time.perf_counter() - started
        digest = hashlib.sha256(b''.join(struct.pack('<I', token) for token in probe)).hexdigest()
        return pp, tg, digest, phases


def engine_hash(site):
    root = Path(site)
    paths = list(root.glob('mlx_lm/**/*.py'))
    for package in ['mlx', 'mlx_metal']:
        for suffix in ['so', 'dylib', 'metallib']:
            paths.extend(root.glob(f'{package}/**/*.{suffix}'))
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
