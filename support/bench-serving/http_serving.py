"""Run the shared concurrent HTTP suite with frozen CLI inputs."""
import argparse
import json
from pathlib import Path
import shlex

import serving_bench
import suite


def run(options, identity):
    settings, cases, _ = suite.load(Path(options['suite']))
    if any(settings[key] != identity['model'][key] for key in ['revision']) or settings['model'] != identity['model']['id']:
        raise ValueError('suite and frozen server model differ')
    command = json.loads(Path(options['server']).read_text())
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError('server command must be a nonempty JSON argv array')
    artifacts = identity['artifacts']
    repository = Path(identity['repository'])
    args = argparse.Namespace(implementation=identity['implementation'], url=options['url'],
        metrics_url=identity.get('metrics_url'), suite=Path(options['suite']),
        json=Path(options['json']), server_command=shlex.join(command),
        server_log=Path(options['json']).with_suffix('.server.log'),
        repository=repository, forja_repository=repository,
        resolved_config=Path(identity['resolved_config']) if identity.get('resolved_config') else None,
        artifact=[(name, Path(item['path'])) for name, item in artifacts.items()],
        weights=[Path(path) for path in identity['weights']], reps=options['reps'],
        warmups=options['warmups'], raw_samples=options['raw_samples'], priming='bounded')
    record = serving_bench.run(args)
    record['server_provenance'] = record['provenance']
    record['provenance'] = {**record['provenance'], **options['provenance']}
    record['server_identity'] = identity
    record['sampling_seed'] = None if all(case['sampling'] == 'greedy' for case in cases) else settings['seed']
    return record
