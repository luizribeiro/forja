#!/usr/bin/env python3
"""Generate deterministic Qwen3-0.6B reference fixtures."""

import argparse
import time
from pathlib import Path
from typing import Any

from fixture_library import generate

PROMPTS = (
    ("short-english", "A quiet forge glows beneath the mountain."),
    (
        "code",
        "def checked_product(values):\n"
        "    result = 1\n"
        "    for value in values:\n"
        "        result *= value\n"
        "    return result\n",
    ),
    (
        "long-paragraph",
        "At dawn the research team crossed the empty courtyard and unlocked "
        "the laboratory. Their task was deliberately ordinary: measure each "
        "instrument, record every assumption, and repeat the experiment until "
        "small discrepancies had an explanation. One engineer calibrated the "
        "sensors while another compared yesterday's notes with the raw data. "
        "They resisted the temptation to smooth an awkward result, because an "
        "unexpected value can reveal more than a perfect chart. By noon they "
        "had traced a drifting measurement to a loose connector hidden behind "
        "the control panel. After replacing it, they ran the full sequence "
        "again with the same inputs and fixed timing. The second record agreed "
        "with the first everywhere except at the repaired channel. Before "
        "leaving, the team saved both records, wrote down the software versions, "
        "labeled the physical samples, and asked a colleague from another group "
        "to reproduce the calculation independently. The work felt slow, but "
        "the resulting evidence was clear enough that anyone could inspect it, "
        "challenge it, and obtain the same answer without relying on memory.",
    ),
    ("single-token", "Hello"),
)

TENSOR_DESCRIPTIONS = {
    "prompt_token_ids": "Tokenized raw prompt, shape [prompt_tokens], I64.",
    "hidden_state_0": "Token embeddings before decoder layer 1, shape [prompt_tokens, 1024], F32.",
    "hidden_state_i": "For 1 <= i < 28, output of decoder layer i before the final model norm, shape [prompt_tokens, 1024], F32.",
    "hidden_state_28": "Output of decoder layer 28 after the final model RMS norm, shape [prompt_tokens, 1024], F32.",
    "prompt_last_logits": "Logits at the last prompt position, shape [151936], F32.",
    "greedy_token_ids": "The 32 autoregressive argmax token ids, shape [32], I64.",
    "greedy_step_logits": "Logits used to select each greedy token, shape [32, 151936], F32.",
}


def validate_qwen_fixture(name: str, token_ids: Any, tensors: dict[str, Any]) -> None:
    """Check assumptions specific to the Qwen3-0.6B fixture set."""
    if name == "single-token" and token_ids.shape[1] != 1:
        raise ValueError(f"single-token prompt encoded as {token_ids.shape[1]} tokens")
    hidden_states = [key for key in tensors if key.startswith("hidden_state_")]
    if len(hidden_states) != 29:
        raise ValueError("Qwen3-0.6B did not return 29 hidden states")


def main() -> None:
    """Parse command-line arguments and generate fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    generate(
        args.model.resolve(),
        args.out.resolve(),
        PROMPTS,
        TENSOR_DESCRIPTIONS,
        validate_fixture=validate_qwen_fixture,
    )
    print(f"completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
