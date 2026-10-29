#!/usr/bin/env python3
"""Generate full SmolLM2-135M-Instruct transformer-f32 fixtures."""

import argparse
from pathlib import Path

from fixture_library import generate
from generate import PROMPTS

TENSORS = {
    "prompt_token_ids": "Raw prompt token ids, [prompt_tokens], I64.",
    "hidden_state_0": "Embedding output, [prompt_tokens, 576], F32.",
    "hidden_state_i": "Decoder i output for 1 <= i < 30, [prompt_tokens, 576], F32.",
    "hidden_state_30": "Final normalized decoder output, [prompt_tokens, 576], F32.",
    "prompt_last_logits": "Last prompt logits, [49152], F32.",
    "greedy_token_ids": "32 autoregressive argmax ids, [32], I64.",
    "greedy_step_logits": "Logits selecting the greedy ids, [32, 49152], F32.",
}


def validate_fixture(name, token_ids, tensors):
    """Require the full architecture and the single-token boundary case."""
    if name == "single-token" and token_ids.shape[1] != 1:
        raise ValueError("single-token prompt must encode as one token")
    if sum(key.startswith("hidden_state_") for key in tensors) != 31:
        raise ValueError("SmolLM2 must return 31 hidden states")


def main():
    """Record deterministic fixtures with library and weight provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    generate(args.model.resolve(), args.out.resolve(), PROMPTS, TENSORS,
             validate_fixture=validate_fixture)


if __name__ == "__main__":
    main()
