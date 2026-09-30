#!/usr/bin/env python3
"""Generate deterministic OLMoE-1B-7B-0924 reference fixtures."""

import argparse
import time
from pathlib import Path

from fixture_library import generate

PROMPTS = (
    ("short-english", "A careful engineer checks every assumption."),
    ("question", "Why does a metal tool expand when heated?"),
    ("continuation", "The safest way to test the result is"),
)

TENSOR_DESCRIPTIONS = {
    "prompt_token_ids": "Tokenized raw prompt, shape [prompt_tokens], I64.",
    "hidden_state_0": "Token embeddings before decoder layer 1, shape [prompt_tokens, 2048], F32.",
    "hidden_state_i": "For 1 <= i < 16, decoder layer i output before the final norm, shape [prompt_tokens, 2048], F32.",
    "hidden_state_16": "Decoder layer 16 output after the final model RMS norm, shape [prompt_tokens, 2048], F32.",
    "router_logits": "Prompt router scores before top-k, shape [16, prompt_tokens, 64], F32.",
    "prompt_last_logits": "Logits at the last prompt position, shape [50304], F32.",
    "greedy_token_ids": "The 32 autoregressive argmax token ids, shape [32], I64.",
    "greedy_step_logits": "Logits used to select each greedy token, shape [32, 50304], F32.",
}


def main() -> None:
    """Parse command-line arguments and generate fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    generate(args.model.resolve(), args.out.resolve(), PROMPTS, TENSOR_DESCRIPTIONS)
    print(f"completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
