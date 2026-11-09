use serde_json::{Value, json};

use crate::{Completion, FinishReason, GenerationError};

/// Builds an OpenAI-shaped non-streaming chat-completion body.
///
/// # Errors
///
/// Returns an error when total token accounting overflows.
pub fn chat_response(
    id: &str,
    created: u64,
    model: &str,
    completion: &Completion,
) -> Result<Value, GenerationError> {
    let message = if completion.tool_calls.is_empty() {
        json!({"role": "assistant", "content": completion.text})
    } else {
        let calls: Vec<Value> = completion
            .tool_calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "id": format!("call_{index}"), "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
        json!({"role": "assistant", "content": (!completion.text.is_empty()).then_some(&completion.text), "tool_calls": calls})
    };
    Ok(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason(completion.finish_reason),
        }],
        "usage": usage(completion)?,
    }))
}

/// Builds an OpenAI-shaped non-streaming text-completion body.
///
/// # Errors
///
/// Returns an error when total token accounting overflows.
pub fn text_response(
    id: &str,
    created: u64,
    model: &str,
    completion: &Completion,
) -> Result<Value, GenerationError> {
    Ok(json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": completion.text,
            "finish_reason": finish_reason(completion.finish_reason),
        }],
        "usage": usage(completion)?,
    }))
}

pub(crate) const fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
    }
}

pub(crate) fn usage(completion: &Completion) -> Result<Value, GenerationError> {
    let prompt = completion.usage.prompt_tokens;
    let generated = completion.usage.completion_tokens;
    let total = prompt
        .checked_add(generated)
        .ok_or_else(|| GenerationError::new("usage token count overflowed"))?;
    Ok(json!({
        "prompt_tokens": prompt,
        "prompt_tokens_details": {"cached_tokens": completion.usage.cached_tokens},
        "completion_tokens": generated,
        "total_tokens": total,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Usage;

    fn completion() -> Completion {
        Completion {
            text: "hello".to_owned(),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 7,
                cached_tokens: 5,
                completion_tokens: 3,
            },
            token_ids: vec![1, 2, 3],
            drafting: crate::DraftMetrics::default(),
        }
    }

    #[test]
    fn serializes_chat_and_text_shapes_with_usage() {
        let chat = chat_response("chatcmpl-1", 2, "served", &completion()).unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hello");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        assert_eq!(chat["usage"]["total_tokens"], 10);
        assert_eq!(chat["usage"]["prompt_tokens_details"]["cached_tokens"], 5);

        let text = text_response("cmpl-1", 2, "served", &completion()).unwrap();
        assert_eq!(text["object"], "text_completion");
        assert_eq!(text["choices"][0]["text"], "hello");
    }

    #[test]
    fn serializes_one_validated_tool_call() {
        let mut completion = completion();
        completion.text.clear();
        completion.finish_reason = FinishReason::ToolCalls;
        completion.tool_calls.push(crate::GeneratedToolCall {
            name: "read_file".to_owned(),
            arguments: r#"{"path":"a.rs"}"#.to_owned(),
        });
        let response = chat_response("chatcmpl-1", 2, "served", &completion).unwrap();
        let choice = &response["choices"][0];
        assert!(choice["message"]["content"].is_null());
        assert_eq!(choice["message"]["tool_calls"][0]["id"], "call_0");
        assert_eq!(choice["finish_reason"], "tool_calls");
    }
    #[test]
    fn serializes_multiple_calls_with_distinct_ids() {
        let mut completion = completion();
        completion.finish_reason = FinishReason::ToolCalls;
        completion.tool_calls = vec![
            crate::GeneratedToolCall {
                name: "lookup".to_owned(),
                arguments: "{}".to_owned()
            };
            2
        ];
        let response = chat_response("id", 0, "model", &completion).unwrap();
        let calls = response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "call_0");
        assert_eq!(calls[1]["id"], "call_1");
        assert_eq!(response["choices"][0]["message"]["content"], "hello");
    }
}
