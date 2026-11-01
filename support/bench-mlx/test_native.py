import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('mlx_native_adapter', Path(__file__).with_name('native.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_protocol_refuses_reordered_relative_or_existing_outputs(tmp_path):
    path = tmp_path/'record.json'
    assert native.request(json.dumps({'id':1,'output':str(path)}),1)['id'] == 1
    for request in [{'id':2,'output':str(path)}, {'id':1,'output':'relative.json'}]:
        with pytest.raises(ValueError):
            native.request(json.dumps(request),1)
    path.write_text('{}')
    with pytest.raises(ValueError, match='new absolute'):
        native.request(json.dumps({'id':1,'output':str(path)}),1)


def test_protocol_identity_has_one_load_and_a_fresh_request_contract(capsys):
    native.emit('READY')
    record = json.loads(capsys.readouterr().out.split(' ',1)[1])
    assert record['protocol'] == 3
    assert record['runner_loads'] == 1 and record['instance_loads'] == {'single':1}
