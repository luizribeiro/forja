"""The text grant retains tensor bytes and excludes vision storage."""
import importlib.util
import json
from pathlib import Path

import pytest


def downloader():
    spec = importlib.util.spec_from_file_location("qwen38_download", Path(__file__).parents[1] / "engines/qwen38/download.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_grant_copies_bytes_and_refuses_changed_output(tmp_path):
    module = downloader()
    shards = [
        ("first.safetensors", {
            "language_model.model.z": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "vision_tower.middle": {"dtype":"F32","shape":[1],"data_offsets":[4,8]},
            "language_model.model.a": {"dtype":"F32","shape":[1],"data_offsets":[8,12]},
        }, b"ztaglostatag"),
        ("second.safetensors", {
            "vision_tower.first": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "language_model.lm_head.weight": {"dtype":"F32","shape":[2],"data_offsets":[4,12]},
        }, b"lostheadDATA"),
    ]
    for name, tensors, payload in shards:
        header = json.dumps(tensors).encode()
        (tmp_path / name).write_bytes(len(header).to_bytes(8,"little") + header + payload)
    names = [name for name, _, _ in shards]
    output = module.text_grant(tmp_path,names,None)
    data = output.read_bytes()
    length = int.from_bytes(data[:8],"little")
    rewritten = json.loads(data[8:8+length])
    expected_tensors = {
        "language_model.lm_head.weight": ([0,8], b"headDATA"),
        "language_model.model.a": ([8,12], b"atag"),
        "language_model.model.z": ([12,16], b"ztag"),
    }
    assert rewritten.keys() == expected_tensors.keys()
    payload = data[8+length:]
    assert payload == b"headDATAatagztag"
    for key, (offsets, tensor_bytes) in expected_tensors.items():
        assert rewritten[key]["data_offsets"] == offsets
        assert payload[slice(*offsets)] == tensor_bytes
    expected = module.digest(output)
    assert module.text_grant(tmp_path,names,expected) == output
    output.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed text weights"):
        module.text_grant(tmp_path,names,expected)
    assert output.read_bytes() == b"changed"


def test_refuses_changed_pinned_input_without_network(tmp_path):
    module = downloader()
    path = tmp_path / "config.json"
    path.write_bytes(b"pinned")
    expected = module.digest(path)
    module.fetch(tmp_path,"unused","unused",{path.name:expected})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="refusing to replace"):
        module.fetch(tmp_path,"unused","unused",{path.name:expected})


@pytest.mark.parametrize("duplicate", [False, True])
def test_invalid_tensor_errors_identify_source_shard_and_tensor(tmp_path, duplicate):
    module = downloader()
    key = "language_model.model.value"
    header = json.dumps({key:{"dtype":"F32","shape":[1],"data_offsets":[0,4 if duplicate else 8]}}).encode()
    names = ["first.safetensors", "second.safetensors"] if duplicate else ["broken.safetensors"]
    for name in names:
        (tmp_path / name).write_bytes(len(header).to_bytes(8,"little") + header + b"data")
    with pytest.raises(ValueError, match=names[-1] + ".*" + key):
        module.text_grant(tmp_path,names,None)
