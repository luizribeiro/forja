use std::{
    collections::VecDeque,
    error::Error,
    fs,
    io::{Read, Write},
    path::Path,
    process::Command,
};

use forja_core::Backend;
use forja_host::{EngineDecode, EngineRunner, SamplingParams};
use tokenizers::Tokenizer;

use crate::{
    args::{Backend as BackendArg, Run},
    engine::{limits, read_token, weights_path},
};

#[cfg(target_os = "macos")]
use crate::engine::metal_graph_replay;

pub(crate) async fn run(options: &Run) -> Result<(), Box<dyn Error>> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    match options.backend {
        BackendArg::Cpu => {
            generate(
                forja_cpu::CpuBackend::new(),
                options,
                &options.engine,
                &mut output,
            )
            .await?;
        }
        BackendArg::Metal => {
            #[cfg(target_os = "macos")]
            generate(
                forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(
                    options.graph_replay,
                ))
                .map_err(|error| format!("cannot create Metal backend: {error}"))?,
                options,
                &options.engine,
                &mut output,
            )
            .await?;
            #[cfg(not(target_os = "macos"))]
            return Err("the Metal backend requires macOS".into());
        }
    }
    Ok(())
}

async fn generate<B, W>(
    backend: B,
    options: &Run,
    component: &Path,
    output: &mut W,
) -> Result<Vec<u32>, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
    W: Write,
{
    let report = generate_report(backend, options, component, output).await?;
    eprintln!(
        "engine memory: live={} bytes, RSS={} bytes",
        report.live_bytes, report.rss_bytes
    );
    Ok(report.tokens)
}

struct GenerationReport {
    tokens: Vec<u32>,
    live_bytes: u64,
    rss_bytes: u64,
}

async fn generate_report<B, W>(
    backend: B,
    options: &Run,
    component: &Path,
    output: &mut W,
) -> Result<GenerationReport, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
    W: Write,
{
    let tokenizer = Tokenizer::from_file(options.model_dir.join("tokenizer.json"))
        .map_err(|error| format!("cannot load tokenizer: {error}"))?;
    let encoding = tokenizer
        .encode(options.prompt.as_str(), false)
        .map_err(|error| format!("cannot tokenize prompt: {error}"))?;
    let prompt = encoding.get_ids().to_vec();
    if prompt.is_empty() {
        return Err("the prompt tokenized to an empty sequence".into());
    }
    let eos = load_eos_token_ids(&options.model_dir)?;
    let mut runner = EngineRunner::new(
        component,
        backend,
        limits(&options.limits)?,
        weights_path(&options.model_dir)?,
    )
    .await?;
    let info = runner.describe().await?;
    validate_generation_request(&tokenizer, &info, prompt.len(), options.max_tokens)?;
    runner
        .load()
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    if options.max_tokens == 0 {
        return generation_report(Vec::new(), &runner);
    }
    let sampling = sampling_params(options)?;
    let prompt_len = u32::try_from(prompt.len())?;
    let first = runner
        .enqueue_decode(EngineDecode {
            tokens: Some(prompt),
            start_pos: 0,
            sampling,
        })
        .await?
        .map_err(|error| format!("engine decode failed: {error:?}"))?;
    let mut stream = tokenizer.decode_stream(true);
    let mut generated = Vec::with_capacity(options.max_tokens);
    let mut pending = VecDeque::from([first]);
    let mut issued = 1_usize;
    let replay_base = runner.metrics().submissions;
    let mut consumed = 0_usize;
    let mut position = prompt_len;
    let generation = async {
        while !pending.is_empty() {
            while pending.len() < 2 && issued < options.max_tokens {
                pending.push_back(
                    runner
                        .enqueue_decode(EngineDecode {
                            tokens: None,
                            start_pos: position,
                            sampling,
                        })
                        .await?
                        .map_err(|error| format!("engine decode failed: {error:?}"))?,
                );
                position = position
                    .checked_add(1)
                    .ok_or("decode position overflowed")?;
                issued += 1;
            }
            let result = pending.pop_front().ok_or("decode queue is empty")?;
            let token = read_token(&runner.read_queued_token(result).await?)?;
            consumed += 1;
            if consumed > 1 {
                let expected = replay_base
                    .checked_add(u64::try_from(consumed - 1)?)
                    .ok_or("submission count overflowed")?;
                runner.wait_for_submissions(expected).await?;
            }
            generated.push(token);
            if eos.contains(&token) {
                discard_speculative_tail(&mut runner, &mut pending).await?;
                break;
            }
            if let Some(text) = stream
                .step(token)
                .map_err(|error| format!("cannot decode token: {error}"))?
            {
                output.write_all(text.as_bytes())?;
                output.flush()?;
            }
        }
        Ok::<(), Box<dyn Error>>(())
    }
    .await;
    if let Err(error) = generation {
        return match discard_speculative_tail(&mut runner, &mut pending).await {
            Ok(()) => Err(error),
            Err(discard) => Err(format!("{error}; speculative drain failed: {discard:?}").into()),
        };
    }
    generation_report(generated, &runner)
}

fn generation_report<B: Backend + Send + Sync + 'static>(
    tokens: Vec<u32>,
    runner: &EngineRunner<B>,
) -> Result<GenerationReport, Box<dyn Error>> {
    Ok(GenerationReport {
        tokens,
        live_bytes: runner.metrics().live_bytes,
        rss_bytes: rss_bytes()?,
    })
}

fn rss_bytes() -> Result<u64, Box<dyn Error>> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    if !output.status.success() {
        return Err("ps failed while reading resident memory".into());
    }
    let kibibytes = String::from_utf8(output.stdout)?.trim().parse::<u64>()?;
    Ok(kibibytes
        .checked_mul(1024)
        .ok_or("RSS byte count overflowed")?)
}

fn sampling_params(options: &Run) -> Result<SamplingParams, Box<dyn Error>> {
    let seed = match (options.seed, options.temperature > 0.0) {
        (Some(seed), _) => seed,
        (None, true) => {
            let mut bytes = [0_u8; 8];
            fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
            let seed = u64::from_ne_bytes(bytes);
            eprintln!("sampling seed: {seed}");
            seed
        }
        (None, false) => 0,
    };
    Ok(SamplingParams {
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        seed,
    })
}

fn validate_generation_request(
    tokenizer: &Tokenizer,
    info: &forja_host::EngineInfo,
    prompt_len: usize,
    max_tokens: usize,
) -> Result<(), Box<dyn Error>> {
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&token| token >= info.vocab)
    {
        return Err("tokenizer contains ids outside the engine vocabulary".into());
    }
    let occupied = prompt_len
        .checked_add(max_tokens.saturating_sub(1))
        .ok_or("requested context length overflowed")?;
    if occupied > usize::try_from(info.max_context)? {
        return Err("prompt and maximum generation exceed the engine context".into());
    }
    Ok(())
}

/// Releases unread speculative outputs after every pipeline exit.
///
/// Speculative decodes after EOS may write KV entries past the emitted tokens. This is safe
/// because a later prefill or step overwrites a position before attending to it. The current
/// runner is invalidated after a discard because its retained feedback token also advanced.
async fn discard_speculative_tail<B>(
    runner: &mut EngineRunner<B>,
    pending: &mut VecDeque<forja_host::EngineDecodeOutput>,
) -> Result<(), forja_host::bindings::l9o::gpu::compute::Error>
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
    failure.map_or(Ok(()), Err)
}

fn load_eos_token_ids(model_dir: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes = fs::read(model_dir.join("generation_config.json"))?;
    let config: serde_json::Value = serde_json::from_slice(&bytes)?;
    eos_token_ids(&config)
}

fn eos_token_ids(config: &serde_json::Value) -> Result<Vec<u32>, Box<dyn Error>> {
    let value = config
        .get("eos_token_id")
        .ok_or("generation config has no eos_token_id")?;
    let values = value
        .as_array()
        .map_or_else(|| vec![value], |ids| ids.iter().collect());
    let ids = values
        .into_iter()
        .map(|id| {
            id.as_u64()
                .ok_or("eos_token_id must contain unsigned integers")
                .and_then(|id| u32::try_from(id).map_err(|_| "eos_token_id exceeds u32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("eos_token_id cannot be empty".into());
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use std::{env, path::PathBuf};

    use golden_fixtures::FixtureDirectory;
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_scalar_and_array_eos_ids() -> Result<(), Box<dyn Error>> {
        assert_eq!(eos_token_ids(&json!({ "eos_token_id": 7 }))?, [7]);
        assert_eq!(eos_token_ids(&json!({ "eos_token_id": [7, 33] }))?, [7, 33]);
        Ok(())
    }

    #[test]
    fn rejects_missing_eos_ids() {
        assert!(eos_token_ids(&json!({})).is_err());
    }

    #[test]
    fn rejects_non_integer_eos_ids() {
        assert!(eos_token_ids(&json!({ "eos_token_id": 1.5 })).is_err());
    }

    #[test]
    fn rejects_an_empty_eos_array() {
        assert!(eos_token_ids(&json!({ "eos_token_id": [] })).is_err());
    }

    #[test]
    fn rejects_eos_ids_over_u32() {
        let overflow = u64::from(u32::MAX) + 1;
        assert!(eos_token_ids(&json!({ "eos_token_id": overflow })).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn short_english_matches_the_golden_continuation() -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let options = Run {
            engine: test_guests::qwen3().to_owned(),
            model_dir: root.join("Qwen3-0.6B"),
            prompt: "A quiet forge glows beneath the mountain.".to_owned(),
            max_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            backend: BackendArg::Metal,
            graph_replay: forja_config::GraphReplay::Tier2,
            limits: forja_config::Limits::default(),
        };
        let expected = FixtureDirectory::open(root.join("golden/qwen3-0.6b"))?
            .prompt("short-english")
            .ok_or("short-english fixture is missing")?
            .greedy_tokens()
            .iter()
            .map(|&token| u32::try_from(token))
            .collect::<Result<Vec<_>, _>>()?;
        let actual = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(generate(
                forja_metal::MetalBackend::new()?,
                &options,
                test_guests::qwen3(),
                &mut Vec::new(),
            ))?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn qwen3_coder_full_model_generates_code() -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let options = Run {
            engine: test_guests::qwen3_coder().to_owned(),
            model_dir: root.join("Qwen3-Coder-30B-A3B-Instruct-4bit"),
            prompt: "fn fibonacci(n: u64) -> u64 {\n".to_owned(),
            max_tokens: 32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: None,
            backend: BackendArg::Metal,
            graph_replay: forja_config::GraphReplay::Tier2,
            limits: forja_config::Limits {
                live_bytes: forja_config::ByteSize::new(64 * 1024 * 1024 * 1024),
                ..forja_config::Limits::default()
            },
        };
        let mut output = Vec::new();
        let report = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(generate_report(
                forja_metal::MetalBackend::new()?,
                &options,
                test_guests::qwen3_coder(),
                &mut output,
            ))?;
        let text = String::from_utf8(output)?;
        eprintln!(
            "Qwen3-Coder full model: live={} bytes, RSS={} bytes, output={text:?}",
            report.live_bytes, report.rss_bytes,
        );
        assert!(!report.tokens.is_empty());
        assert!(text.trim().len() >= 8);
        assert!(
            text.chars()
                .any(|character| character.is_ascii_alphanumeric())
        );
        assert!(!text.contains('\u{fffd}'));
        Ok(())
    }
}
