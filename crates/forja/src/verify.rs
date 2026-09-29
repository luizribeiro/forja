use std::{error::Error, path::Path};

use forja_core::Backend;
use forja_host::{EngineRunner, EngineStep, EngineTensor};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, BF16_LOGIT_KL_TOLERANCE, FixtureDirectory, LOGIT_KL_TOLERANCE,
    decode_f32_le, mean_logit_kl_divergence, normwise_relative_error, sha256_file,
};

use crate::args::{Backend as BackendArg, Precision, Verify};
use crate::engine::{argmax, limits};

const F32_HIDDEN_STATE_TOLERANCE: f64 = 2e-2;
const BF16_TOP1_PERCENT: u64 = 95;

#[derive(Clone, Copy, Default)]
struct DecodeMetrics {
    total_kl: f64,
    maximum_kl: f64,
    agreement: u32,
    steps: u32,
}

impl DecodeMetrics {
    fn mean_kl(self) -> f64 {
        if self.steps == 0 {
            f64::INFINITY
        } else {
            self.total_kl / f64::from(self.steps)
        }
    }

    fn include(&mut self, other: Self) {
        self.total_kl += other.total_kl;
        self.maximum_kl = self.maximum_kl.max(other.maximum_kl);
        self.agreement += other.agreement;
        self.steps += other.steps;
    }
}

#[derive(Clone, Copy)]
enum DecodeMode {
    FreeRunning,
    TeacherForced,
}

#[derive(Clone, Copy, Default)]
struct VerificationSummary {
    maximum_layer_error: f64,
    teacher_forced: DecodeMetrics,
    free_running: DecodeMetrics,
}

struct PromptVerification {
    maximum_layer_error: f64,
    layers_passed: bool,
    passed: bool,
    teacher_forced: DecodeMetrics,
    free_running: DecodeMetrics,
}

struct DecodeInput<'a> {
    prompt: &'a str,
    expected_tokens: &'a [i64],
    reference_logits: &'a [f32],
    prompt_tokens: u32,
    logits: Vec<f32>,
    vocab: usize,
    steps: usize,
}

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
        &options.engine,
        decode_steps,
        true,
    )
    .await
    .map(|_| ())
}
async fn run_with_component(
    options: &Verify,
    fixtures: &FixtureDirectory,
    weights: &Path,
    component: &Path,
    decode_steps: usize,
    enforce_tolerances: bool,
) -> Result<VerificationSummary, Box<dyn Error>> {
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
) -> Result<VerificationSummary, Box<dyn Error>>
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
    let mut summary = VerificationSummary::default();
    println!("prompt\tlayer\trelative-error\tresult");
    for fixture in prompts {
        let prompt = verify_prompt(
            &mut runner,
            fixture,
            options.precision,
            usize::try_from(info.vocab)?,
            decode_steps,
        )
        .await?;
        summary.maximum_layer_error = summary.maximum_layer_error.max(prompt.maximum_layer_error);
        summary.teacher_forced.include(prompt.teacher_forced);
        summary.free_running.include(prompt.free_running);
        passed &= match options.precision {
            Precision::F32 => prompt.passed,
            Precision::Bf16 => prompt.layers_passed,
        };
    }
    if options.precision == Precision::Bf16 {
        let teacher_passed = summary.teacher_forced.mean_kl() <= BF16_LOGIT_KL_TOLERANCE
            && top1_passes(summary.teacher_forced);
        passed &= teacher_passed;
        println!(
            "all-prompts\tteacher-mean-kl={:.8e}\tteacher-max-kl={:.8e}\ttop-1={}/{}\t{}",
            summary.teacher_forced.mean_kl(),
            summary.teacher_forced.maximum_kl,
            summary.teacher_forced.agreement,
            summary.teacher_forced.steps,
            if passed { "pass" } else { "FAIL" }
        );
    }
    if passed || !enforce_tolerances {
        Ok(summary)
    } else {
        Err("verification failed".into())
    }
}

async fn verify_prompt<B>(
    runner: &mut EngineRunner<B>,
    fixture: &golden_fixtures::PromptFixture,
    precision: Precision,
    vocab: usize,
    decode_steps: usize,
) -> Result<PromptVerification, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let tokens = fixture
        .prompt_ids()
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    let output = runner
        .step(EngineStep {
            tokens: tokens.clone(),
            start_pos: 0,
            taps: true,
        })
        .await?
        .map_err(|error| format!("engine step failed: {error:?}"))?;
    let logits = decode_f32_le(&runner.read(&output.logits).await?)?;
    let layer_tolerance = match precision {
        Precision::F32 => F32_HIDDEN_STATE_TOLERANCE,
        Precision::Bf16 => BF16_HIDDEN_STATE_TOLERANCE,
    };
    let (maximum_layer_error, first_failing) =
        compare_layers(runner, fixture, &output.taps, layer_tolerance).await?;
    let input = DecodeInput {
        prompt: fixture.name(),
        expected_tokens: fixture.greedy_tokens(),
        reference_logits: fixture.greedy_step_logits().values(),
        prompt_tokens: u32::try_from(fixture.prompt_ids().len())?,
        logits,
        vocab,
        steps: decode_steps,
    };
    match precision {
        Precision::F32 => {
            verify_f32_decode(runner, fixture, input, maximum_layer_error, first_failing).await
        }
        Precision::Bf16 => {
            verify_bf16_decode(runner, input, tokens, maximum_layer_error, first_failing).await
        }
    }
}

async fn compare_layers<B>(
    runner: &mut EngineRunner<B>,
    fixture: &golden_fixtures::PromptFixture,
    taps: &[EngineTensor],
    tolerance: f64,
) -> Result<(f64, Option<usize>), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let mut first_failing = None;
    let mut maximum = 0.0_f64;
    for (index, tap) in taps.iter().enumerate() {
        let values = decode_f32_le(&runner.read(tap).await?)?;
        let reference = fixture
            .hidden_state(index + 1)
            .ok_or("hidden-state fixture is missing")?;
        let error = normwise_relative_error(reference.values(), &values)?;
        maximum = maximum.max(error);
        let passed = error <= tolerance;
        if !passed && first_failing.is_none() {
            first_failing = Some(index + 1);
        }
        println!(
            "{}\t{}\t{error:.8e}\t{}",
            fixture.name(),
            index + 1,
            if passed { "pass" } else { "FAIL" }
        );
    }
    Ok((maximum, first_failing))
}

async fn verify_f32_decode<B>(
    runner: &mut EngineRunner<B>,
    fixture: &golden_fixtures::PromptFixture,
    input: DecodeInput<'_>,
    maximum_layer_error: f64,
    first_failing: Option<usize>,
) -> Result<PromptVerification, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let prompt_kl =
        mean_logit_kl_divergence(fixture.prompt_logits().values(), &input.logits, input.vocab)?;
    let free_running = decode(runner, input, DecodeMode::FreeRunning, true).await?;
    let passed = first_failing.is_none()
        && prompt_kl <= LOGIT_KL_TOLERANCE
        && free_running.mean_kl() <= LOGIT_KL_TOLERANCE
        && free_running.agreement == free_running.steps;
    println!(
        "{}\tmax-layer={maximum_layer_error:.8e}\tfirst-failing={}\tprompt-kl={prompt_kl:.8e}\tfree-mean-kl={:.8e}\tfree-max-kl={:.8e}\ttop-1={}/{}\t{}",
        fixture.name(),
        layer_name(first_failing),
        free_running.mean_kl(),
        free_running.maximum_kl,
        free_running.agreement,
        free_running.steps,
        if passed { "pass" } else { "FAIL" }
    );
    Ok(PromptVerification {
        maximum_layer_error,
        layers_passed: first_failing.is_none(),
        passed,
        teacher_forced: DecodeMetrics::default(),
        free_running,
    })
}

async fn verify_bf16_decode<B>(
    runner: &mut EngineRunner<B>,
    input: DecodeInput<'_>,
    prompt_tokens: Vec<u32>,
    maximum_layer_error: f64,
    first_failing: Option<usize>,
) -> Result<PromptVerification, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let prompt = input.prompt;
    let expected_tokens = input.expected_tokens;
    let reference_logits = input.reference_logits;
    let prompt_length = input.prompt_tokens;
    let vocab = input.vocab;
    let steps = input.steps;
    let teacher_forced = decode(runner, input, DecodeMode::TeacherForced, true).await?;
    let prompt_passed = first_failing.is_none()
        && teacher_forced.mean_kl() <= BF16_LOGIT_KL_TOLERANCE
        && top1_passes(teacher_forced);
    println!(
        "{}\tmax-layer={maximum_layer_error:.8e}\tfirst-failing={}\tteacher-mean-kl={:.8e}\tteacher-max-kl={:.8e}\ttop-1={}/{}\t{}",
        prompt,
        layer_name(first_failing),
        teacher_forced.mean_kl(),
        teacher_forced.maximum_kl,
        teacher_forced.agreement,
        teacher_forced.steps,
        if prompt_passed { "pass" } else { "FAIL" }
    );
    let output = runner
        .step(EngineStep {
            tokens: prompt_tokens,
            start_pos: 0,
            taps: false,
        })
        .await?
        .map_err(|error| format!("engine step failed: {error:?}"))?;
    let free_input = DecodeInput {
        prompt,
        expected_tokens,
        reference_logits,
        prompt_tokens: prompt_length,
        logits: decode_f32_le(&runner.read(&output.logits).await?)?,
        vocab,
        steps,
    };
    let free_running = decode(runner, free_input, DecodeMode::FreeRunning, false).await?;
    println!(
        "{}\tfree-running-mean-kl={:.8e}\tfree-running-max-kl={:.8e}\ttop-1={}/{}\tinformational",
        prompt,
        free_running.mean_kl(),
        free_running.maximum_kl,
        free_running.agreement,
        free_running.steps
    );
    Ok(PromptVerification {
        maximum_layer_error,
        layers_passed: first_failing.is_none(),
        passed: prompt_passed,
        teacher_forced,
        free_running,
    })
}

fn layer_name(layer: Option<usize>) -> String {
    layer.map_or_else(|| "-".to_owned(), |layer| layer.to_string())
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
    input: DecodeInput<'_>,
    mode: DecodeMode,
    report_steps: bool,
) -> Result<DecodeMetrics, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let DecodeInput {
        prompt,
        expected_tokens,
        reference_logits,
        prompt_tokens,
        mut logits,
        vocab,
        steps,
    } = input;
    if steps == 0 || steps > expected_tokens.len() {
        return Err("decode step count is outside the fixture".into());
    }
    let mut metrics = DecodeMetrics {
        steps: u32::try_from(steps)?,
        ..DecodeMetrics::default()
    };
    if report_steps {
        println!("prompt\tstep\tlogits-kl\texpected\tactual\tresult");
    }
    for (step, &expected) in expected_tokens.iter().take(steps).enumerate() {
        let start = step
            .checked_mul(vocab)
            .ok_or("decode fixture offset overflowed")?;
        let end = start
            .checked_add(vocab)
            .ok_or("decode fixture offset overflowed")?;
        let reference = reference_logits
            .get(start..end)
            .ok_or("decode logits fixture is incomplete")?;
        let kl = mean_logit_kl_divergence(reference, &logits, vocab)?;
        metrics.total_kl += kl;
        metrics.maximum_kl = metrics.maximum_kl.max(kl);
        let actual = argmax(&logits)?;
        let expected = u32::try_from(expected)?;
        let token_matches = actual == expected;
        metrics.agreement += u32::from(token_matches);
        if report_steps {
            println!(
                "{prompt}\t{step}\t{kl:.8e}\t{expected}\t{actual}\t{}",
                if token_matches { "pass" } else { "FAIL" }
            );
        }
        if step + 1 < steps {
            let start_pos = prompt_tokens
                .checked_add(u32::try_from(step)?)
                .ok_or("decode position overflowed")?;
            let output = runner
                .step(EngineStep {
                    tokens: vec![match mode {
                        DecodeMode::FreeRunning => actual,
                        DecodeMode::TeacherForced => expected,
                    }],
                    start_pos,
                    taps: false,
                })
                .await?
                .map_err(|error| format!("decode step failed: {error:?}"))?;
            logits = decode_f32_le(&runner.read(&output.logits).await?)?;
        }
    }
    Ok(metrics)
}

fn top1_passes(metrics: DecodeMetrics) -> bool {
    metrics.steps > 0
        && u64::from(metrics.agreement) * 100 >= u64::from(metrics.steps) * BF16_TOP1_PERCENT
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
        run_model_test(
            BackendArg::Cpu,
            Precision::F32,
            vec!["single-token".to_owned()],
            2,
        )
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen_verification() -> Result<(), Box<dyn Error>> {
        run_model_test(BackendArg::Metal, Precision::F32, Vec::new(), 32)
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_qwen_fusion_profiles() -> Result<(), Box<dyn Error>> {
        let options = model_options(
            BackendArg::Metal,
            Precision::F32,
            vec!["short-english".to_owned()],
        )?;
        let fixtures = FixtureDirectory::open(&options.fixtures)?;
        let weights = verify_model_hash(&options, &fixtures)?;
        let components = [
            test_guests::qwen3_residual_norm(),
            test_guests::qwen3_qk_norm_rope(),
            test_guests::qwen3_silu_mul(),
            test_guests::qwen3_final_norm(),
            test_guests::qwen3_all_fusions(),
        ];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for component in components {
            runtime.block_on(run_with_component(
                &options, &fixtures, &weights, component, 32, true,
            ))?;
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_bf16_qwen_verification() -> Result<(), Box<dyn Error>> {
        let options = model_options(BackendArg::Metal, Precision::Bf16, Vec::new())?;
        let fixtures = FixtureDirectory::open(&options.fixtures)?;
        let weights = verify_model_hash(&options, &fixtures)?;
        let summary = tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_component(
                &options,
                &fixtures,
                &weights,
                test_guests::qwen3_bf16(),
                32,
                true,
            ))?;
        assert!(summary.maximum_layer_error <= BF16_HIDDEN_STATE_TOLERANCE);
        assert!(summary.teacher_forced.mean_kl() <= BF16_LOGIT_KL_TOLERANCE);
        assert!(top1_passes(summary.teacher_forced));
        Ok(())
    }

    #[test]
    fn teacher_forcing_uses_reference_tokens() -> Result<(), Box<dyn Error>> {
        let root = temporary_directory("teacher-forcing")?;
        let weights = root.join("weights.safetensors");
        fs::write(&weights, empty_safetensors())?;
        let result = tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(async {
                let mut runner = EngineRunner::new(
                    test_guests::engine_smoke(),
                    forja_cpu::CpuBackend::new(),
                    limits(),
                    &weights,
                )
                .await?;
                runner.load().await??;
                decode(
                    &mut runner,
                    DecodeInput {
                        prompt: "tiny",
                        expected_tokens: &[0, 1, 2],
                        reference_logits: &[
                            4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0,
                        ],
                        prompt_tokens: 1,
                        logits: vec![1.0, 1.0, 1.0, 4.0],
                        vocab: 4,
                        steps: 3,
                    },
                    DecodeMode::TeacherForced,
                    false,
                )
                .await
            });
        fs::remove_dir_all(root)?;
        let metrics = result?;
        assert_eq!(metrics.agreement, 2);
        assert_eq!(metrics.steps, 3);
        Ok(())
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
            engine: PathBuf::from("unused.wasm"),
            model_dir: model,
            fixtures: fixtures_path.clone(),
            backend: BackendArg::Cpu,
            precision: Precision::F32,
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
        precision: Precision,
        prompts: Vec<String>,
        decode_steps: usize,
    ) -> Result<(), Box<dyn Error>> {
        let options = model_options(backend, precision, prompts)?;
        tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_steps(&options, decode_steps))
    }

    fn model_options(
        backend: BackendArg,
        precision: Precision,
        prompts: Vec<String>,
    ) -> Result<Verify, Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        Ok(Verify {
            engine: match precision {
                Precision::F32 => test_guests::qwen3(),
                Precision::Bf16 => test_guests::qwen3_bf16(),
            }
            .to_owned(),
            model_dir: root.join("Qwen3-0.6B"),
            fixtures: root.join("golden/qwen3-0.6b"),
            backend,
            precision,
            prompts,
        })
    }

    fn temporary_directory(test: &str) -> Result<PathBuf, Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = env::temp_dir().join(format!("forja-verify-{test}-{nonce}"));
        fs::create_dir(&root)?;
        Ok(root)
    }

    fn empty_safetensors() -> Vec<u8> {
        let mut header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        bytes
    }
}
