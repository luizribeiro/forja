import io
import json

import http_client
import pytest


def test_stream_retains_server_counts_and_actual_tool_history():
    client = http_client.OpenAIClient("http://127.0.0.1:1", "test")
    events = [
        {"choices": [{"delta": {"content": "hello", "tool_calls": [
            {"index": 0, "id": "call", "function": {"name": "lookup", "arguments": '{"key":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"value"}'}}]}}]},
        {"usage": {"prompt_tokens": 7, "completion_tokens": 3,
                   "prompt_tokens_details": {"cached_tokens": 1}}},
    ]
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    response = io.BytesIO(body + b"data: [DONE]\n\n")
    result = client.read_stream(response, 0, 0)
    assert (result["content_events"], result["output_tokens"]) == (2, 3)
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert result["assistant_message"] == {"role": "assistant", "content": "hello", "tool_calls": [
        {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"value"}'}}]}


def test_stream_rejects_tool_call_gaps():
    with pytest.raises(ValueError, match="index"):
        http_client.merge_message({}, {"tool_calls": [{"index": 1}]})
