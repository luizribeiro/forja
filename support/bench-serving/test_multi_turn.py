from pathlib import Path

import multi_turn

class Client:
    def __init__(self):
        self.histories = []

    def chat(self, messages, output_tokens):
        self.histories.append([dict(message) for message in messages])
        turn = len(self.histories)
        return {
            "ttft_seconds": float(turn),
            "itl_seconds": [0.01, 0.01],
            "prompt_tokens": turn * 100,
            "output_tokens": output_tokens,
            "cached_tokens": 0 if turn == 1 else (turn - 1) * 100,
            "output": "generated",
            "error": None,
            "e2e_seconds": float(turn) + .02,
        }

def test_committed_suite_is_a_twelve_turn_screen() -> None:
    path = Path(__file__).resolve().parents[2] / "bench/suites/multi-turn.toml"
    suite, digest = multi_turn.load_suite(path)
    assert (suite["repetitions"], len(suite["tool_output_tokens"])) == (5, 12)
    assert (min(suite["tool_output_tokens"]), max(suite["tool_output_tokens"])) == (1536, 2560)
    assert len(digest) == 64


def test_conversation_grows_with_scripted_assistant_content() -> None:
    suite = {
        "tool_output_tokens": [2, 3],
        "max_context": 1000,
        "output_tokens": 4,
        "scripted_assistant_tokens": 2,
    }
    client = Client()
    samples, prompts = multi_turn.run_conversation(client, suite, "sample-0")
    assert len(samples) == len(prompts) == 2
    assert [message["role"] for message in client.histories[1]] == [
        "system", "user", "assistant", "tool", "user", "assistant", "tool"
    ]
    assert client.histories[1][2]["content"] == " result result"
    reports = multi_turn.turn_reports([samples], suite["tool_output_tokens"])
    assert reports[1]["cached_tokens"]["median"] == 100
    assert reports[0]["cache_phase"] == "cold"
    assert reports[1]["cache_phase"] == "cached"
    assert reports[1]["total_context_tokens"]["median"] == 204
    assert reports[0]["decode_tokens_per_second"]["median"] == 100.0


def test_raw_turn_windows_keep_whole_sessions_as_independent_units():
    suite = {"tool_output_tokens": [2], "max_context": 1000,
             "output_tokens": 4, "scripted_assistant_tokens": 2}
    samples, _ = multi_turn.run_conversation(Client(), suite, "sample-0", raw_samples=True)
    report = multi_turn.turn_reports([samples], [2])[0]["samples"][0]
    assert report["windows"]["ttft"][1] - report["windows"]["ttft"][0] == 1.
    assert abs(report["windows"]["decode"][1] - report["windows"]["decode"][0] - .02) < 1e-6
