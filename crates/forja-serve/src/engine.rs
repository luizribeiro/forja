use std::{error::Error, fmt, fs::File, io::Read};

use forja_config::HostConfig;
use forja_core::Backend;
use forja_host::{EngineDecode, EngineDecodeOutput, EngineInfo, EngineRunner, SamplingParams};
use tokenizers::Tokenizer;

use crate::request::{validate_chat_input, validate_prompt};
use crate::{ChatCompletionRequest, ChatFormat, ResolvedParams, serving_context_ceiling};

/// A failure while preparing or running a generation.
#[derive(Debug)]
pub struct GenerationError {
    message: String,
}

impl GenerationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for GenerationError {}

/// Device-resident first-token selection and its validated request policy.
pub struct PrefilledPrompt {
    /// First selected token, still resident on the backend.
    pub first: EngineDecodeOutput,
    /// Validated generation policy.
    pub params: ResolvedParams,
    /// Rendered prompt token count.
    pub prompt_tokens: u32,
    /// Engine sampling parameters used by every decode step.
    pub sampling: SamplingParams,
}

trait PromptRenderer {
    fn render_prompt(&mut self, messages: &str) -> Result<String, GenerationError>;
}

impl PromptRenderer for ChatFormat {
    fn render_prompt(&mut self, messages: &str) -> Result<String, GenerationError> {
        self.render(messages, "null", r#"{"add_generation_prompt":true}"#)
            .map_err(|error| GenerationError::new(format!("chat format failed: {error}")))?
            .map_err(GenerationError::new)
    }
}

/// One loaded engine, tokenizer, and model-declared chat format.
pub struct ServingEngine<B: Backend + Send + Sync + 'static> {
    runner: EngineRunner<B>,
    tokenizer: Tokenizer,
    chat_format: ChatFormat,
    config: HostConfig,
    info: EngineInfo,
    max_context: u32,
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    /// Joins already-loaded serving artifacts after checking their shared boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error when the profile or tokenizer exceeds the engine contract.
    pub fn new(
        runner: EngineRunner<B>,
        tokenizer: Tokenizer,
        chat_format: ChatFormat,
        config: HostConfig,
        info: EngineInfo,
        profile_max_context: u32,
    ) -> Result<Self, GenerationError> {
        let max_context = serving_context_ceiling(
            info.max_context,
            profile_max_context,
            config.request.max_prompt_tokens,
        )
        .map_err(|error| GenerationError::new(error.to_string()))?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&token| token >= info.vocab)
        {
            return Err(GenerationError::new(
                "tokenizer contains ids outside the engine vocabulary",
            ));
        }
        Ok(Self {
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            max_context,
        })
    }

    /// Renders, tokenizes, validates, and prefills one chat request at position zero.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, chat-format failure, or engine failure.
    pub async fn prefill_chat(
        &mut self,
        request: &ChatCompletionRequest,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let chat_info = self.chat_format.info().clone();
        let (tokens, prompt_tokens, params) = prepare_chat_prompt(
            request,
            &self.config,
            &chat_info,
            self.max_context,
            self.info.vocab,
            &self.tokenizer,
            &mut self.chat_format,
        )?;
        let sampling = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            seed: sampling_seed(params.seed, params.temperature)?,
        };
        let first = self
            .runner
            .enqueue_decode(EngineDecode {
                tokens: Some(tokens),
                start_pos: 0,
                sampling,
            })
            .await
            .map_err(|error| GenerationError::new(format!("engine prefill failed: {error}")))?
            .map_err(|error| GenerationError::new(format!("engine prefill failed: {error:?}")))?;
        Ok(PrefilledPrompt {
            first,
            params,
            prompt_tokens,
            sampling,
        })
    }
}

fn prepare_chat_prompt<R: PromptRenderer>(
    request: &ChatCompletionRequest,
    config: &HostConfig,
    chat_info: &crate::ChatInfo,
    max_context: u32,
    vocab: u32,
    tokenizer: &Tokenizer,
    renderer: &mut R,
) -> Result<(Vec<u32>, u32, ResolvedParams), GenerationError> {
    let params = validate_chat_input(request, config, chat_info, vocab)
        .map_err(|error| GenerationError::new(error.to_string()))?;
    let messages = serde_json::to_string(&request.messages)
        .map_err(|error| GenerationError::new(format!("cannot encode messages: {error}")))?;
    let prompt_text = renderer.render_prompt(&messages)?;
    let rendered_bytes = u64::try_from(prompt_text.len())
        .map_err(|_| GenerationError::new("rendered prompt size exceeds u64"))?;
    if rendered_bytes > config.request.max_text_bytes.get() {
        return Err(GenerationError::new(
            "rendered prompt exceeds the configured byte ceiling",
        ));
    }
    let tokens = tokenizer
        .encode(prompt_text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| GenerationError::new(format!("cannot tokenize prompt: {error}")))?;
    let prompt_tokens = u32::try_from(tokens.len())
        .map_err(|_| GenerationError::new("prompt token count exceeds u32"))?;
    if tokens.is_empty() {
        return Err(GenerationError::new(
            "rendered prompt tokenized to an empty sequence",
        ));
    }
    let params = validate_prompt(params, prompt_tokens, max_context)
        .map_err(|error| GenerationError::new(error.to_string()))?;
    Ok((tokens, prompt_tokens, params))
}

fn sampling_seed(requested: Option<u64>, temperature: f32) -> Result<u64, GenerationError> {
    if let Some(seed) = requested {
        return Ok(seed);
    }
    if temperature == 0.0 {
        return Ok(0);
    }
    let mut bytes = [0_u8; 8];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| GenerationError::new(format!("cannot read sampling entropy: {error}")))?;
    Ok(u64::from_ne_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicUsize};

    struct CountingRenderer(Arc<AtomicUsize>);

    impl PromptRenderer for CountingRenderer {
        fn render_prompt(&mut self, _: &str) -> Result<String, GenerationError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok("rendered".to_owned())
        }
    }

    #[test]
    fn invalid_message_count_never_calls_chat_format() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut renderer = CountingRenderer(Arc::clone(&calls));
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let mut config = HostConfig::default();
        config.request.max_messages = 0;
        let request = ChatCompletionRequest {
            model: "test".to_owned(),
            messages: vec![crate::Message {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            params: crate::CompletionParams::default(),
        };
        let info = crate::ChatInfo {
            stop_token_ids: Vec::new(),
            sampling: crate::SamplingDefaults::default(),
        };

        let result = prepare_chat_prompt(
            &request,
            &config,
            &info,
            32_768,
            151_936,
            &tokenizer,
            &mut renderer,
        );

        assert!(result.is_err());
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn greedy_sampling_needs_no_entropy() {
        assert_eq!(sampling_seed(None, 0.0).unwrap(), 0);
        assert_eq!(sampling_seed(Some(17), 0.7).unwrap(), 17);
    }
}
