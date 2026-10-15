use std::{collections::VecDeque, error::Error, fmt, fs::File, io::Read};

use forja_config::HostConfig;
use forja_core::Backend;
use forja_host::{EngineDecode, EngineDecodeOutput, EngineInfo, EngineRunner, SamplingParams};
use tokenizers::Tokenizer;

use crate::request::{validate_chat_input, validate_prompt};
use crate::{
    ChatCompletionRequest, ChatFormat, ResolvedParams, StopMatcher, serving_context_ceiling,
};

/// A failure while preparing or running a generation.
#[derive(Debug)]
pub struct GenerationError {
    message: String,
    kind: ErrorKind,
}

impl GenerationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Internal,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Invalid,
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Unavailable,
        }
    }

    pub(crate) const fn kind(&self) -> ErrorKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ErrorKind {
    Invalid,
    Internal,
    Unavailable,
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
            .map_err(GenerationError::invalid)
    }
}

/// Why token generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// A model-declared token or request stop string matched.
    Stop,
    /// The validated output-token limit was reached.
    Length,
}

/// Token counts for one completed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u32,
    /// Sampled tokens, including a token that caused a stop.
    pub completion_tokens: u32,
}

/// Fully decoded output from one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Completion {
    /// Decoded text before any matched stop.
    pub text: String,
    /// Why generation ended.
    pub finish_reason: FinishReason,
    /// Request token counts.
    pub usage: Usage,
}

/// One loaded engine, tokenizer, and model-declared chat format.
pub struct ServingEngine<B: Backend + Send + Sync + 'static> {
    model: String,
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
        model: String,
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
            model,
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            max_context,
        })
    }

    /// Returns the only model identifier accepted by this engine.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
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
            .map_err(|error| {
                GenerationError::unavailable(format!("engine prefill failed: {error}"))
            })?
            .map_err(|error| {
                GenerationError::unavailable(format!("engine prefill failed: {error:?}"))
            })?;
        Ok(PrefilledPrompt {
            first,
            params,
            prompt_tokens,
            sampling,
        })
    }

    /// Generates a complete chat response with two decode calls in flight.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, backend synchronization, or detokenization fails.
    pub async fn complete_chat(
        &mut self,
        request: &ChatCompletionRequest,
    ) -> Result<Completion, GenerationError> {
        let prefill = self.prefill_chat(request).await?;
        self.decode(prefill).await
    }

    async fn decode(&mut self, prefill: PrefilledPrompt) -> Result<Completion, GenerationError> {
        let mut pending = VecDeque::from([prefill.first]);
        let mut issued = 1_u32;
        let mut consumed = 0_u32;
        let mut position = prefill.prompt_tokens;
        let replay_base = self.runner.metrics().submissions;
        let mut decoder = self.tokenizer.decode_stream(true);
        let mut matcher = StopMatcher::new(&prefill.params.stops);
        let mut text = String::new();
        let generation = async {
            while let Some(output) = pending.pop_front() {
                while pending.len() < 2 && issued < prefill.params.max_tokens {
                    pending.push_back(
                        self.runner
                            .enqueue_decode(EngineDecode {
                                tokens: None,
                                start_pos: position,
                                sampling: prefill.sampling,
                            })
                            .await
                            .map_err(|error| {
                                GenerationError::unavailable(format!(
                                    "engine decode failed: {error}"
                                ))
                            })?
                            .map_err(|error| {
                                GenerationError::unavailable(format!(
                                    "engine decode failed: {error:?}"
                                ))
                            })?,
                    );
                    position = position
                        .checked_add(1)
                        .ok_or_else(|| GenerationError::new("decode position overflowed"))?;
                    issued += 1;
                }
                let bytes = self
                    .runner
                    .read_queued_token(output)
                    .await
                    .map_err(|error| {
                        GenerationError::unavailable(format!(
                            "selected-token read failed: {error:?}"
                        ))
                    })?;
                let token = read_token(&bytes)?;
                consumed += 1;
                if consumed > 1 {
                    let expected = replay_base
                        .checked_add(u64::from(consumed - 1))
                        .ok_or_else(|| GenerationError::new("submission count overflowed"))?;
                    self.runner
                        .wait_for_submissions(expected)
                        .await
                        .map_err(|error| {
                            GenerationError::unavailable(format!(
                                "engine synchronization failed: {error}"
                            ))
                        })?;
                }
                if self.chat_format.info().stop_token_ids.contains(&token) {
                    return Ok(FinishReason::Stop);
                }
                if let Some(piece) = decoder.step(token).map_err(|error| {
                    GenerationError::new(format!("cannot decode token: {error}"))
                })? {
                    let output = matcher.push(piece.as_bytes()).map_err(|error| {
                        GenerationError::new(format!("decoded invalid UTF-8: {error}"))
                    })?;
                    text.push_str(&output.text);
                    if output.matched {
                        return Ok(FinishReason::Stop);
                    }
                }
            }
            let output = matcher
                .finish()
                .map_err(|error| GenerationError::new(format!("decoded invalid UTF-8: {error}")))?;
            text.push_str(&output.text);
            Ok(FinishReason::Length)
        }
        .await;
        let discard = discard_pending(&mut self.runner, &mut pending).await;
        let finish_reason = match (generation, discard) {
            (Ok(reason), Ok(())) => reason,
            (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
        };
        Ok(Completion {
            text,
            finish_reason,
            usage: Usage {
                prompt_tokens: prefill.prompt_tokens,
                completion_tokens: consumed,
            },
        })
    }
}

fn read_token(bytes: &[u8]) -> Result<u32, GenerationError> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| GenerationError::new("selected token is not one u32"))?;
    Ok(u32::from_le_bytes(bytes))
}

async fn discard_pending<B>(
    runner: &mut EngineRunner<B>,
    pending: &mut VecDeque<EngineDecodeOutput>,
) -> Result<(), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let mut failure = None;
    for output in pending.drain(..) {
        if let Err(error) = runner.discard_queued_decode(output).await
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), |error| {
        Err(GenerationError::unavailable(format!(
            "speculative decode cleanup failed: {error:?}"
        )))
    })
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
        .map_err(|error| GenerationError::invalid(error.to_string()))?;
    let messages = serde_json::to_string(&request.messages)
        .map_err(|error| GenerationError::new(format!("cannot encode messages: {error}")))?;
    let prompt_text = renderer.render_prompt(&messages)?;
    let rendered_bytes = u64::try_from(prompt_text.len())
        .map_err(|_| GenerationError::invalid("rendered prompt size exceeds u64"))?;
    if rendered_bytes > config.request.max_text_bytes.get() {
        return Err(GenerationError::invalid(
            "rendered prompt exceeds the configured byte ceiling",
        ));
    }
    let tokens = tokenizer
        .encode(prompt_text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| GenerationError::new(format!("cannot tokenize prompt: {error}")))?;
    let prompt_tokens = u32::try_from(tokens.len())
        .map_err(|_| GenerationError::invalid("prompt token count exceeds u32"))?;
    if tokens.is_empty() {
        return Err(GenerationError::invalid(
            "rendered prompt tokenized to an empty sequence",
        ));
    }
    let params = validate_prompt(params, prompt_tokens, max_context)
        .map_err(|error| GenerationError::invalid(error.to_string()))?;
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
