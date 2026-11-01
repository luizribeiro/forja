"""Frozen inputs and CPU token checks for HTTP conversations."""
import copy
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load(path):
    identity = json.loads(Path(path).read_text())
    if not {'model', 'artifacts', 'effective_config', 'tokenizer_path'} <= identity.keys():
        raise ValueError('incomplete HTTP benchmark identity')
    if not {'tokenizer', 'tokenizer_config', 'weights', 'server'} <= identity['artifacts'].keys():
        raise ValueError('identity must bind tokenizer, weights and server artifacts')
    for name, artifact in identity['artifacts'].items():
        if sha256(artifact['path']) != artifact['sha256']:
            raise ValueError(f'frozen HTTP artifact mismatch: {name}')
    return identity


class TokenCheck:
    """Render exactly the pinned tokenizer's template before admitting a request."""
    def __init__(self, identity):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(identity['tokenizer_path'], local_files_only=True)
        self.options = identity.get('chat_template_kwargs', {})

    def prompt(self, messages, output_tokens, ceiling):
        normalized = copy.deepcopy(messages)
        for message in normalized:
            for call in message.get('tool_calls', []):
                function = call['function']
                if isinstance(function.get('arguments'), str):
                    function['arguments'] = json.loads(function['arguments'])
        text = self.tokenizer.apply_chat_template(normalized, tokenize=False,
            add_generation_prompt=True, **self.options)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) + output_tokens > ceiling:
            raise ValueError('conversation exceeds its audited whole-command context ceiling')
        return {'prompt_tokens': len(tokens), 'maximum_context': len(tokens) + output_tokens,
                'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'tokens_sha256': hashlib.sha256(b''.join(t.to_bytes(4, 'little') for t in tokens)).hexdigest()}
