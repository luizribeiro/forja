use std::{error::Error, time::Duration};

use forja_core::Backend;
use forja_host::{EngineRunner, EngineStep, Limits};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, FixtureDirectory, LOGIT_KL_TOLERANCE, mean_logit_kl_divergence,
    normwise_relative_error,
};

use crate::args::{Backend as BackendArg, Verify};

pub(crate) async fn run(options: &Verify) -> Result<(), Box<dyn Error>> {
    let fixtures = FixtureDirectory::open(&options.fixtures)?;
    match options.backend {
        BackendArg::Cpu => verify(forja_cpu::CpuBackend::new(), options, &fixtures).await,
        BackendArg::Metal => {
            #[cfg(target_os = "macos")]
            {
                let backend = forja_metal::MetalBackend::new()
                    .map_err(|error| format!("cannot create Metal backend: {error}"))?;
                verify(backend, options, &fixtures).await
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
) -> Result<(), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let weights = options.model_dir.join("model.safetensors");
    let mut runner = EngineRunner::new(test_guests::qwen3(), backend, limits(), weights).await?;
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
        let logits = decode_f32(&runner.read(&output.logits).await?)?;
        let mut first_failing = None;
        for (index, tap) in output.taps.iter().enumerate() {
            let values = decode_f32(&runner.read(tap).await?)?;
            let reference = fixture
                .hidden_state(index + 1)
                .ok_or("hidden-state fixture is missing")?;
            let error = normwise_relative_error(reference.values(), &values)?;
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
        let prompt_passed = first_failing.is_none() && logits_kl <= LOGIT_KL_TOLERANCE;
        passed &= prompt_passed;
        println!(
            "{}\tfirst-failing={}\tprompt-logits-kl={logits_kl:.8e}\t{}",
            fixture.name(),
            first_failing.map_or_else(|| "-".to_owned(), |layer| layer.to_string()),
            if prompt_passed { "pass" } else { "FAIL" }
        );
    }
    if passed {
        Ok(())
    } else {
        Err("verification failed".into())
    }
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

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err("tensor contains a partial f32 value".into());
    }
    Ok(values
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

const fn limits() -> Limits {
    Limits::new(
        8 * 1024 * 1024 * 1024,
        4,
        1_000_000_000,
        20_000,
        1024 * 1024 * 1024,
    )
    .with_command_limits(4_096, u64::MAX)
    .with_guest_call_timeout(Duration::from_secs(300))
    .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600))
}
