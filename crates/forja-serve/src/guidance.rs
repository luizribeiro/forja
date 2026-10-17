//! Bounded grammar compilation and token-mask execution.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::Duration,
};

use llguidance::{
    CancellationHandle, Constraint, ParserFactory,
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{ApproximateTokEnv, INVALID_TOKEN, TokEnv, TokRxInfo, TokTrie},
};
use tokenizers::Tokenizer;
use tokio::sync::oneshot;

const COMPILE_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_GRAMMAR_BYTES: usize = 256 * 1024;

enum Job {
    Compile {
        id: u64,
        grammar: String,
        cancelled: Arc<AtomicBool>,
        reply: oneshot::Sender<Result<CancellationHandle, String>>,
    },
    Mask {
        id: u64,
        reply: oneshot::Sender<Result<Option<Vec<u32>>, String>>,
    },
    Commit {
        id: u64,
        token: u32,
        reply: oneshot::Sender<Result<(), String>>,
    },
    #[cfg(test)]
    LexerStates {
        id: u64,
        reply: oneshot::Sender<Result<usize, String>>,
    },
    Remove(u64),
}

/// One-worker, one-queued-job executor for llguidance.
pub struct GuidanceWorker {
    jobs: SyncSender<Job>,
    next_id: AtomicU64,
}

/// One grammar constraint owned by a [`GuidanceWorker`].
#[derive(Debug)]
pub struct GuidanceSession {
    id: u64,
    jobs: SyncSender<Job>,
    cancellation: CancellationHandle,
}

impl GuidanceWorker {
    /// Builds the tokenizer trie once and starts the bounded worker.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible tokenizer data or thread creation failure.
    pub fn new(tokenizer: &Tokenizer, vocab: u32, eos: &[u32]) -> Result<Self, String> {
        let json = tokenizer
            .to_string(false)
            .map_err(|error| format!("cannot serialize tokenizer: {error}"))?;
        let json = serde_json::from_str(&json)
            .map_err(|error| format!("cannot parse tokenizer data: {error}"))?;
        let mut tokens = llguidance::token_bytes_from_tokenizer_json(&json)
            .map_err(|error| format!("cannot build tokenizer bytes: {error}"))?;
        let vocab_len = usize::try_from(vocab)
            .map_err(|_| "engine vocabulary does not fit usize".to_owned())?;
        if tokens.len() > vocab_len || eos.iter().any(|&token| token >= vocab) {
            return Err("tokenizer ids exceed the engine vocabulary".to_owned());
        }
        tokens.resize(vocab_len, Vec::new());
        let primary_eos = eos.first().copied().unwrap_or(INVALID_TOKEN);
        let mut trie = TokTrie::from(&TokRxInfo::new(vocab, primary_eos), &tokens);
        if !eos.is_empty() {
            trie = trie.with_eos_tokens(eos);
        }
        let environment: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let mut factory = ParserFactory::new_simple(&environment)
            .map_err(|error| format!("cannot build grammar factory: {error}"))?;
        *factory.limits_mut() = parser_limits();
        factory.quiet();
        let mask_words = vocab
            .checked_add(31)
            .ok_or_else(|| "vocabulary mask size overflowed".to_owned())?
            / 32;
        let mut valid_tokens = vec![
            0_u32;
            usize::try_from(mask_words).map_err(|_| {
                "vocabulary mask size does not fit usize".to_owned()
            })?
        ];
        for &token in tokenizer.get_vocab(true).values() {
            if token < vocab {
                let word = usize::try_from(token / 32)
                    .map_err(|_| "token id does not fit usize".to_owned())?;
                valid_tokens[word] |= 1_u32 << (token % 32);
            }
        }
        let (jobs, receiver) = sync_channel(1);
        thread::Builder::new()
            .name("forja-llguidance".to_owned())
            .spawn(move || run_worker(&receiver, &factory, mask_words, &valid_tokens))
            .map_err(|error| format!("cannot start grammar worker: {error}"))?;
        Ok(Self {
            jobs,
            next_id: AtomicU64::new(1),
        })
    }

    /// Compiles one Lark grammar under the 500 ms watchdog.
    ///
    /// # Errors
    ///
    /// Returns an error when the grammar is too large, invalid, too complex, or times out.
    pub async fn compile(&self, grammar: String) -> Result<GuidanceSession, String> {
        if grammar.len() > MAX_GRAMMAR_BYTES {
            return Err("grammar exceeds the configured byte ceiling".to_owned());
        }
        let id = self
            .next_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
            .map_err(|_| "grammar session id space is exhausted".to_owned())?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (reply, response) = oneshot::channel();
        self.jobs
            .try_send(Job::Compile {
                id,
                grammar,
                cancelled: Arc::clone(&cancelled),
                reply,
            })
            .map_err(|_| "grammar worker is busy".to_owned())?;
        let cancellation = match tokio::time::timeout(COMPILE_TIMEOUT, response).await {
            Ok(Ok(result)) => result?,
            Ok(Err(_)) => return Err("grammar worker stopped".to_owned()),
            Err(_) => {
                cancelled.store(true, Ordering::Release);
                return Err("grammar compilation exceeded 500 ms".to_owned());
            }
        };
        Ok(GuidanceSession {
            id,
            jobs: self.jobs.clone(),
            cancellation,
        })
    }
}

impl GuidanceSession {
    /// Computes the next packed allow mask, or `None` when the grammar is complete.
    pub async fn mask(&self) -> Result<Option<Vec<u32>>, String> {
        let (reply, response) = oneshot::channel();
        self.jobs
            .try_send(Job::Mask { id: self.id, reply })
            .map_err(|_| "grammar worker is busy".to_owned())?;
        response
            .await
            .map_err(|_| "grammar worker stopped".to_owned())?
    }

    /// Commits one sampled token.
    pub async fn commit(&self, token: u32) -> Result<(), String> {
        let (reply, response) = oneshot::channel();
        self.jobs
            .try_send(Job::Commit {
                id: self.id,
                token,
                reply,
            })
            .map_err(|_| "grammar worker is busy".to_owned())?;
        response
            .await
            .map_err(|_| "grammar worker stopped".to_owned())?
    }

    #[cfg(test)]
    async fn lexer_states(&self) -> Result<usize, String> {
        let (reply, response) = oneshot::channel();
        self.jobs
            .try_send(Job::LexerStates { id: self.id, reply })
            .map_err(|_| "grammar worker is busy".to_owned())?;
        response
            .await
            .map_err(|_| "grammar worker stopped".to_owned())?
    }

    /// Requests cooperative cancellation without waiting for the worker.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
}

impl Drop for GuidanceSession {
    fn drop(&mut self) {
        self.cancellation.cancel();
        let _ = self.jobs.try_send(Job::Remove(self.id));
    }
}

fn parser_limits() -> ParserLimits {
    ParserLimits {
        max_items_in_row: 2_000,
        initial_lexer_fuel: 1_000_000,
        step_lexer_fuel: 200_000,
        step_max_items: 50_000,
        max_lexer_states: 100_000,
        max_grammar_size: 200_000,
        verbose_errors: false,
        ..ParserLimits::default()
    }
}

fn run_worker(
    jobs: &std::sync::mpsc::Receiver<Job>,
    factory: &ParserFactory,
    mask_words: u32,
    valid_tokens: &[u32],
) {
    let mut constraints = HashMap::<u64, Constraint>::new();
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Compile {
                id,
                grammar,
                cancelled,
                reply,
            } => {
                constraints.clear();
                let result = factory
                    .create_parser(TopLevelGrammar::from_lark(grammar))
                    .map(Constraint::new)
                    .map_err(|error| format!("grammar compilation failed: {error}"))
                    .and_then(|mut constraint| {
                        let cancellation = constraint.parser.enable_cancellation();
                        if cancelled.load(Ordering::Acquire) {
                            cancellation.cancel();
                            return Err("grammar compilation cancelled".to_owned());
                        }
                        constraints.insert(id, constraint);
                        Ok(cancellation)
                    });
                if reply.send(result).is_err() {
                    constraints.remove(&id);
                }
            }
            Job::Mask { id, reply } => {
                let result = constraints
                    .get_mut(&id)
                    .ok_or_else(|| "grammar session is not active".to_owned())
                    .and_then(|constraint| {
                        let step = constraint
                            .compute_mask()
                            .map_err(|error| format!("mask computation failed: {error}"))?;
                        if step.is_stop() {
                            return Ok(None);
                        }
                        let mask = step.sample_mask.as_ref().ok_or_else(|| {
                            "grammar requested unsupported forced tokens".to_owned()
                        })?;
                        let mask_words = usize::try_from(mask_words).unwrap_or(usize::MAX);
                        let packed = mask.as_slice();
                        if packed.len() < mask_words
                            || packed[mask_words..].iter().any(|&word| word != 0)
                        {
                            return Err("grammar produced an invalid token mask".to_owned());
                        }
                        let packed: Vec<u32> = packed[..mask_words]
                            .iter()
                            .zip(valid_tokens)
                            .map(|(&grammar, &valid)| grammar & valid)
                            .collect();
                        if packed.iter().all(|&word| word == 0) {
                            return Err("grammar produced an empty token mask".to_owned());
                        }
                        Ok(Some(packed))
                    });
                let _ = reply.send(result);
            }
            Job::Commit { id, token, reply } => {
                let result = constraints
                    .get_mut(&id)
                    .ok_or_else(|| "grammar session is not active".to_owned())
                    .and_then(|constraint| {
                        let result = constraint
                            .commit_token(Some(token))
                            .map_err(|error| format!("token commitment failed: {error}"))?;
                        if result.backtrack != 0 || result.ff_tokens != [token] {
                            return Err(
                                "grammar returned unexpected backtracking or committed tokens"
                                    .to_owned(),
                            );
                        }
                        Ok(())
                    });
                let _ = reply.send(result);
            }
            #[cfg(test)]
            Job::LexerStates { id, reply } => {
                let result = constraints
                    .get(&id)
                    .ok_or_else(|| "grammar session is not active".to_owned())
                    .map(|constraint| constraint.parser.parser.lexer_stats().num_states);
                let _ = reply.send(result);
            }
            Job::Remove(id) => {
                constraints.remove(&id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf, time::Instant};

    use tokenizers::{Tokenizer, decoders::byte_level::ByteLevel, models::bpe::BPE};

    use super::{GuidanceWorker, MAX_GRAMMAR_BYTES};

    fn tokenizer() -> Tokenizer {
        let vocab = [
            ("a".to_owned(), 0),
            ("b".to_owned(), 1),
            ("<eos>".to_owned(), 2),
        ];
        let model = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
    }

    #[tokio::test]
    async fn compiles_masks_and_commits_on_the_bounded_worker() {
        let worker = GuidanceWorker::new(&tokenizer(), 3, &[2]).unwrap();
        let session = worker.compile("start: \"a\"".to_owned()).await.unwrap();
        let mask = session.mask().await.unwrap().unwrap();
        assert_eq!(mask, [1]);
        session.commit(0).await.unwrap();
        assert!(session.mask().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn trims_zero_sentinel_storage_at_a_mask_word_boundary() {
        let worker = GuidanceWorker::new(&tokenizer(), 32, &[2]).unwrap();
        let session = worker.compile("start: \"a\"".to_owned()).await.unwrap();
        assert_eq!(session.mask().await.unwrap().unwrap(), [1]);
    }

    #[tokio::test]
    async fn refuses_grammar_before_queueing_oversized_input() {
        let worker = GuidanceWorker::new(&tokenizer(), 3, &[2]).unwrap();
        let error = worker
            .compile("a".repeat(MAX_GRAMMAR_BYTES + 1))
            .await
            .unwrap_err();
        assert!(error.contains("byte ceiling"));
    }

    #[tokio::test]
    #[ignore = "requires FORJA_MODELS and records grammar benchmark data"]
    async fn profiles_qwen_tool_grammar() {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").unwrap());
        let tokenizer =
            Tokenizer::from_file(root.join("Qwen3-Coder-30B-A3B-Instruct-4bit/tokenizer.json"))
                .unwrap();
        let mut format = crate::ChatFormat::load(
            test_guests::qwen3_coder_chat_format(),
            crate::ChatFormatLimits::default(),
        )
        .unwrap();
        let tools = serde_json::json!([{"type":"function","function":{
            "name":"read_file","description":"Read one source file","parameters":{
                "type":"object","properties":{"path":{"type":"string","enum":["src/main.rs"]}},
                "required":["path"],"additionalProperties":false
            }
        }}]);
        let grammar = format
            .grammar(
                &tools.to_string(),
                r#"{"type":"function","function":{"name":"read_file"}}"#,
                "null",
            )
            .unwrap()
            .unwrap();
        let worker = GuidanceWorker::new(&tokenizer, 151_936, &[151_645, 151_643]).unwrap();
        let compile_started = Instant::now();
        let session = worker.compile(grammar.grammar).await.unwrap();
        let compile = compile_started.elapsed();
        let mut masks = Vec::new();
        loop {
            let mask_started = Instant::now();
            let Some(mask) = session.mask().await.unwrap() else {
                break;
            };
            masks.push(mask_started.elapsed());
            let (word, bits) = mask
                .iter()
                .copied()
                .enumerate()
                .find(|(_, bits)| *bits != 0)
                .unwrap();
            let token = u32::try_from(word).unwrap() * 32 + bits.trailing_zeros();
            session.commit(token).await.unwrap();
        }
        masks.sort_unstable();
        let percentile = |percent: usize| masks[(masks.len() * percent).div_ceil(100) - 1];
        eprintln!(
            "Qwen grammar compile {compile:?}; lexer states {}; masks {}; mask p50/p90/p99 {:?}/{:?}/{:?}",
            session.lexer_states().await.unwrap(),
            masks.len(),
            percentile(50),
            percentile(90),
            percentile(99)
        );
    }
}
