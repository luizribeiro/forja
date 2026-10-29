"""Full-layer and tokenization assumptions of the SmolLM2 fixtures."""

import pytest

from smollm2 import validate_fixture


class Tokens:
    shape = (1, 1)


def test_requires_all_layers_and_single_token():
    tensors = {f"hidden_state_{index}": None for index in range(31)}
    validate_fixture("single-token", Tokens(), tensors)
    with pytest.raises(ValueError, match="31"):
        validate_fixture("single-token", Tokens(), {})
    tokens = Tokens()
    tokens.shape = (1, 7)
    with pytest.raises(ValueError, match="one token"):
        validate_fixture("single-token", tokens, tensors)
