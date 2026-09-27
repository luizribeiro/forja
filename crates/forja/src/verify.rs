use std::{error::Error, path::Path};

use forja_core::Backend;
use forja_host::{EngineRunner, EngineStep};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, LOGIT_KL_TOLERANCE, decode_f32_le,
    mean_logit_kl_divergence, normwise_relative_error, sha256_file,
};

use crate::args::{Backend as BackendArg, Verify};
use crate::engine::{argmax, limits};

pub(crate) async fn run(options: &Verify) -> Result<(), Box<dyn Error>> {
    run_with_steps(options, 32).await
}

async fn run_with_steps(options: &Verify, decode_steps: usize) -> Result<(), Box<dyn Error>> {
    let fixtures = FixtureDirectory::open(&options.fixtures)?;
    let weights = verify_model_hash(options, &fixtures)?;
    run_with_component(
        options,
        &fixtures,
        &weights,
        test_guests::qwen3(),
        decode_steps,
        true,
    )
    .await
}

async fn run_with_component(
    options: &Verify,
    fixtures: &FixtureDirectory,
    weights: &Path,
    component: &Path,
    decode_steps: usize,
    enforce_tolerances: bool,
) -> Result<(), Box<dyn Error>> {
    match options.backend {
        BackendArg::Cpu => {
            verify(
                forja_cpu::CpuBackend::new(),
                options,
                fixtures,
                weights,
                component,
                decode_steps,
                enforce_tolerances,
            )
            .await
        }
        BackendArg::Metal => {
            #[cfg(target_os = "macos")]
            {
                let backend = forja_metal::MetalBackend::new()
                    .map_err(|error| format!("cannot create Metal backend: {error}"))?;
                verify(
                    backend,
                    options,
                    fixtures,
                    weights,
                    component,
                    decode_steps,
                    enforce_tolerances,
                )
                .await
            }
            #[cfg(not(target_os = "macos"))]
            Err("the Metal backend requires macOS".into())
        }
    }
}

async fn verify<B>(
    backend: B,
    options: &Verify,
    fixtures: &FixtureDirectory,
    weights: &Path,
    component: &Path,
    decode_steps: usize,
    enforce_tolerances: bool,
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let mut runner = EngineRunner::new(component, backend, limits(), weights).await?;
    let info = runner.describe().await?;
    if info.vocab != 151_936
        || info.max_context != 4_096
        || info.tap_layers != (1..=28).collect::<Vec<_>>()
    {
        return Err("Qwen3 engine metadata is incompatible with the verifier".into());
    }
    runner
        .load()
        .await?
        .map_err(|error| format!("engine load failed: {error:?}"))?;
    let prompts = selected_prompts(fixtures, &options.prompts)?;
    let mut passed = true;
    println!("prompt\tlayer\trelative-error\tresult");
    for fixture in prompts {
        let tokens = fixture
            .prompt_ids()
            .iter()
            .map(|&token| u32::try_from(token))
            .collect::<Result<Vec<_>, _>>()?;
        let output = runner
            .step(EngineStep {
                tokens,
                start_pos: 0,
                taps: true,
            })
            .await?
            .map_err(|error| format!("engine step failed: {error:?}"))?;
        let logits = decode_f32_le(&runner.read(&output.logits).await?)?;
        let mut first_failing = None;
        let mut maximum_layer_error = 0.0_f64;
        for (index, tap) in output.taps.iter().enumerate() {
            let values = decode_f32_le(&runner.read(tap).await?)?;
            let reference = fixture
                .hidden_state(index + 1)
                .ok_or("hidden-state fixture is missing")?;
            let error = normwise_relative_error(reference.values(), &values)?;
            maximum_layer_error = maximum_layer_error.max(error);
            let layer_passed = error <= BF16_HIDDEN_STATE_TOLERANCE;
            if !layer_passed && first_failing.is_none() {
                first_failing = Some(index + 1);
            }
            if layer_passed {
                println!("{}\t{}\t{error:.8e}\tpass", fixture.name(), index + 1);
            } else {
                println!("{}\t{}\t{error:.8e}\tFAIL", fixture.name(), index + 1);
            }
        }
        let logits_kl = mean_logit_kl_divergence(
            fixture.prompt_logits().values(),
            &logits,
            usize::try_from(info.vocab)?,
        )?;
        let prompt_tokens = u32::try_from(fixture.prompt_ids().len())?;
        let (decode_kl, agreement) = decode(
            &mut runner,
            fixture,
            prompt_tokens,
            logits,
            usize::try_from(info.vocab)?,
            decode_steps,
        )
        .await?;
        let prompt_passed = first_failing.is_none()
            && logits_kl <= LOGIT_KL_TOLERANCE
            && decode_kl <= LOGIT_KL_TOLERANCE
            && agreement == decode_steps;
        passed &= prompt_passed;
        println!(
            "{}\tmax-layer={maximum_layer_error:.8e}\tfirst-failing={}\tprompt-kl={logits_kl:.8e}\tdecode-kl={decode_kl:.8e}\ttokens={agreement}/{decode_steps}\t{}",
            fixture.name(),
            first_failing.map_or_else(|| "-".to_owned(), |layer| layer.to_string()),
            if prompt_passed { "pass" } else { "FAIL" }
        );
    }
    if passed || !enforce_tolerances {
        Ok(())
    } else {
        Err("verification failed".into())
    }
}

fn verify_model_hash(
    options: &Verify,
    fixtures: &FixtureDirectory,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let weights = options.model_dir.join("model.safetensors");
    let actual = sha256_file(&weights)?;
    let expected = fixtures.model_sha256();
    if actual != expected {
        return Err(format!(
            "model SHA-256 mismatch: model has {actual}, fixtures require {expected}"
        )
        .into());
    }
    Ok(weights)
}

async fn decode<B>(
    runner: &mut EngineRunner<B>,
    fixture: &golden_fixtures::PromptFixture,
    prompt_tokens: u32,
    mut logits: Vec<f32>,
    vocab: usize,
    steps: usize,
) -> Result<(f64, usize), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    if steps == 0 || steps > fixture.greedy_tokens().len() {
        return Err("decode step count is outside the fixture".into());
    }
    let mut total_kl = 0.0;
    let mut agreement = 0;
    println!("prompt\tstep\tlogits-kl\texpected\tactual\tresult");
    for step in 0..steps {
        let start = step
            .checked_mul(vocab)
            .ok_or("decode fixture offset overflowed")?;
        let end = start
            .checked_add(vocab)
            .ok_or("decode fixture offset overflowed")?;
        let reference = fixture
            .greedy_step_logits()
            .values()
            .get(start..end)
            .ok_or("decode logits fixture is incomplete")?;
        let kl = mean_logit_kl_divergence(reference, &logits, vocab)?;
        total_kl += kl;
        let actual = argmax(&logits)?;
        let expected = u32::try_from(fixture.greedy_tokens()[step])?;
        let token_matches = actual == expected;
        agreement += usize::from(token_matches);
        println!(
            "{}\t{step}\t{kl:.8e}\t{expected}\t{actual}\t{}",
            fixture.name(),
            if token_matches { "pass" } else { "FAIL" }
        );
        if step + 1 < steps {
            let start_pos = prompt_tokens
                .checked_add(u32::try_from(step)?)
                .ok_or("decode position overflowed")?;
            let output = runner
                .step(EngineStep {
                    tokens: vec![actual],
                    start_pos,
                    taps: false,
                })
                .await?
                .map_err(|error| format!("decode step failed: {error:?}"))?;
            logits = decode_f32_le(&runner.read(&output.logits).await?)?;
        }
    }
    Ok((total_kl / f64::from(u32::try_from(steps)?), agreement))
}

fn selected_prompts<'a>(
    fixtures: &'a FixtureDirectory,
    names: &[String],
) -> Result<Vec<&'a golden_fixtures::PromptFixture>, Box<dyn Error>> {
    if names.is_empty() {
        return Ok(fixtures.prompts().iter().collect());
    }
    names
        .iter()
        .map(|name| {
            fixtures
                .prompt(name)
                .ok_or_else(|| format!("fixture prompt {name:?} does not exist").into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn cpu_qwen_verification() -> Result<(), Box<dyn Error>> {
        run_model_test(BackendArg::Cpu, vec!["single-token".to_owned()], 2)
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen_verification() -> Result<(), Box<dyn Error>> {
        run_model_test(BackendArg::Metal, Vec::new(), 32)
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_bf16_qwen_measurement() -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let options = Verify {
            model_dir: root.join("Qwen3-0.6B"),
            fixtures: root.join("golden/qwen3-0.6b"),
            backend: BackendArg::Metal,
            prompts: vec!["short-english".to_owned()],
        };
        let fixtures = FixtureDirectory::open(&options.fixtures)?;
        let weights = verify_model_hash(&options, &fixtures)?;
        tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_component(
                &options,
                &fixtures,
                &weights,
                test_guests::qwen3_bf16(),
                32,
                false,
            ))
    }

    #[test]
    fn rejects_a_model_hash_that_differs_from_the_manifest() -> Result<(), Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = env::temp_dir().join(format!("forja-verify-hash-{nonce}"));
        let model = root.join("model");
        let fixtures_path = root.join("fixtures");
        fs::create_dir_all(&model)?;
        fs::create_dir(&fixtures_path)?;
        fs::write(model.join("model.safetensors"), b"not a model")?;
        let expected = "0".repeat(64);
        fs::write(
            fixtures_path.join("manifest.json"),
            format!(
                "{{\"schema_version\":1,\"model\":{{\"sha256\":\"{expected}\"}},\"prompts\":[]}}"
            ),
        )?;
        let options = Verify {
            model_dir: model,
            fixtures: fixtures_path.clone(),
            backend: BackendArg::Cpu,
            prompts: Vec::new(),
        };
        let fixtures = FixtureDirectory::open(fixtures_path)?;
        let actual = sha256_file(options.model_dir.join("model.safetensors"))?;
        let error = verify_model_hash(&options, &fixtures)
            .unwrap_err()
            .to_string();
        assert!(error.contains(&actual));
        assert!(error.contains(&expected));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn run_model_test(
        backend: BackendArg,
        prompts: Vec<String>,
        decode_steps: usize,
    ) -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let options = Verify {
            model_dir: root.join("Qwen3-0.6B"),
            fixtures: root.join("golden/qwen3-0.6b"),
            backend,
            prompts,
        };
        tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_steps(&options, decode_steps))
    }
}
