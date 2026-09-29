use std::{error::Error, fs, io::Write, path::Path};

use forja_core::Backend;
use forja_host::{EngineDecode, EngineRunner};
use tokenizers::Tokenizer;

use crate::{
    args::{Backend as BackendArg, Run},
    engine::{limits, read_token},
};

pub(crate) async fn run(options: &Run) -> Result<(), Box<dyn Error>> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    match options.backend {
        BackendArg::Cpu => {
            generate(
                forja_cpu::CpuBackend::new(),
                options,
                test_guests::qwen3(),
                &mut output,
            )
            .await?;
        }
        BackendArg::Metal => {
            #[cfg(target_os = "macos")]
            generate(
                forja_metal::MetalBackend::new()
                    .map_err(|error| format!("cannot create Metal backend: {error}"))?,
                options,
                test_guests::qwen3(),
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
        limits(),
        options.model_dir.join("model.safetensors"),
    )
    .await?;
    let info = runner.describe().await?;
    if tokenizer
        .get_vocab(true)
        .values()
        .any(|&token| token >= info.vocab)
    {
        return Err("tokenizer contains ids outside the engine vocabulary".into());
    }
    let occupied = prompt
        .len()
        .checked_add(options.max_tokens.saturating_sub(1))
        .ok_or("requested context length overflowed")?;
    if occupied > usize::try_from(info.max_context)? {
        return Err("prompt and maximum generation exceed the engine context".into());
    }
    runner
        .load()
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    if options.max_tokens == 0 {
        return Ok(Vec::new());
    }
    let prompt_len = u32::try_from(prompt.len())?;
    let mut result = runner
        .decode(EngineDecode {
            tokens: Some(prompt),
            start_pos: 0,
        })
        .await?
        .map_err(|error| format!("engine decode failed: {error:?}"))?;
    let mut stream = tokenizer.decode_stream(true);
    let mut generated = Vec::with_capacity(options.max_tokens);
    for index in 0..options.max_tokens {
        let token = read_token(&runner.read(&result.token).await?)?;
        generated.push(token);
        if eos.contains(&token) {
            break;
        }
        if let Some(text) = stream
            .step(token)
            .map_err(|error| format!("cannot decode token: {error}"))?
        {
            output.write_all(text.as_bytes())?;
            output.flush()?;
        }
        if index + 1 < options.max_tokens {
            result = runner
                .decode(EngineDecode {
                    tokens: None,
                    start_pos: prompt_len
                        .checked_add(u32::try_from(index)?)
                        .ok_or("decode position overflowed")?,
                })
                .await?
                .map_err(|error| format!("engine decode failed: {error:?}"))?;
        }
    }
    Ok(generated)
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
            model_dir: root.join("Qwen3-0.6B"),
            prompt: "A quiet forge glows beneath the mountain.".to_owned(),
            max_tokens: 32,
            backend: BackendArg::Metal,
        };
        let expected = FixtureDirectory::open(root.join("golden/qwen3-0.6b"))?
            .prompt("short-english")
            .ok_or("short-english fixture is missing")?
            .greedy_tokens()
            .iter()
            .map(|&token| u32::try_from(token))
            .collect::<Result<Vec<_>, _>>()?;
        let actual = tokio::runtime::Builder::new_current_thread()
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
}
