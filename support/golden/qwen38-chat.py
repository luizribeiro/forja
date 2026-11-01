"""Record CPU-only render references from the pinned Transformers chat template."""

import copy
import hashlib
import json
import os
from pathlib import Path

import tokenizers
import transformers
from transformers import AutoTokenizer

TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


def normalized(messages):
    result = copy.deepcopy(messages)
    for message in result:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return result


def main():
    model = Path(os.environ["FORJA_MODELS"]) / "Qwen3.8-27B-4bit"
    output = Path(__file__).resolve().parents[2] / "chat-formats/qwen38/tests/fixtures"
    template = (model / "chat_template.jinja").read_bytes()
    assert hashlib.sha256(template).hexdigest() == TEMPLATE_SHA256
    assert transformers.__version__ == "5.17.0"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = json.loads((output / "render.json").read_text())
    for case in cases:
        case["rendered"] = tokenizer.apply_chat_template(
            normalized(case["messages"]), tools=case["tools"], tokenize=False, **case["options"]
        )
    (output / "render.json").write_text(
        "[\n" + ",\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n]\n"
    )
    history = next(case for case in cases if case["name"] == "tool-history")
    text = history["rendered"]
    begin = text.index("<tool_call>\n", text.index("<|im_start|>assistant\n"))
    end = text.index("<|im_end|>", begin)
    calls = [call["function"] for call in normalized(history["messages"])[1]["tool_calls"]]
    (output / "calls.json").write_text(
        json.dumps({"wire": text[begin:end], "calls": calls}, ensure_ascii=False) + "\n"
    )
    reference = {
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        "template_sha256": TEMPLATE_SHA256,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "recording": "CPU apply_chat_template(tokenize=False); decode argument strings; preserve insertion order; no inference",
    }
    (output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "output": str(output)}))


if __name__ == "__main__":
    main()
