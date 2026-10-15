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
    Ok(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion.text},
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
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 7,
                completion_tokens: 3,
            },
        }
    }

    #[test]
    fn serializes_chat_and_text_shapes_with_usage() {
        let chat = chat_response("chatcmpl-1", 2, "served", &completion()).unwrap();
        assert_eq!(chat["choices"][0]["message"]["content"], "hello");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        assert_eq!(chat["usage"]["total_tokens"], 10);

        let text = text_response("cmpl-1", 2, "served", &completion()).unwrap();
        assert_eq!(text["object"], "text_completion");
        assert_eq!(text["choices"][0]["text"], "hello");
    }
}
