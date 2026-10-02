use std::{error::Error, path::Path};

use forja_core::Backend;
use forja_host::{EngineRunner, EngineStep, EngineTensor};
use golden_fixtures::{
    BF16_HIDDEN_STATE_TOLERANCE, BF16_LOGIT_KL_TOLERANCE, FixtureDirectory, LOGIT_KL_TOLERANCE,
    decode_f32_le, mean_logit_kl_divergence, normwise_relative_error, sha256_file,
};

use crate::args::{Backend as BackendArg, Precision, Verify};
use crate::engine::{argmax, limits, validate_engine};

#[cfg(target_os = "macos")]
use crate::engine::metal_graph_replay;

const F32_HIDDEN_STATE_TOLERANCE: f64 = 2e-2;
const BF16_TOP1_PERCENT: u64 = 95;
const ROUTER_TOP_K: usize = 8;
const ROUTER_NEAR_TIE: f32 = 1.0e-3;

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

struct PromptChecks {
    maximum_layer_error: f64,
    first_failing: Option<usize>,
    routers_passed: bool,
    first_router_mismatch: Option<String>,
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
    validate_engine(
        &options.engine,
        &options.model_dir,
        options.backend,
        &options.scratch,
    )?;
    let fixtures = FixtureDirectory::open(&options.fixtures)?;
    fixtures.require_complete_model_outputs()?;
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
                let backend = forja_metal::MetalBackend::with_graph_replay(metal_graph_replay(
                    options.graph_replay,
                ))
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
    let mut runner =
        EngineRunner::new(component, backend, limits(&options.limits)?, weights).await?;
    let info = runner.describe().await?;
    let first = fixtures
        .prompts()
        .first()
        .ok_or("fixtures contain no prompts")?;
    let layers = first.hidden_states().len() - 1;
    let router_layers = first.router_logits().map_or(0, |router| router.shape()[0]);
    let expected_taps = (1..=u32::try_from(layers)?).collect::<Vec<_>>();
    let expected_routers = (1..=u32::try_from(router_layers)?).collect::<Vec<_>>();
    if usize::try_from(info.vocab)? != first.prompt_logits().values().len()
        || !info.tap_layers.starts_with(&expected_taps)
        || !info.router_layers.starts_with(&expected_routers)
    {
        return Err("engine metadata is incompatible with the fixtures".into());
    }
    runner
        .load_with_config(Some(u32::try_from(layers)?))
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
    let (routers_passed, first_router_mismatch) =
        compare_routers(runner, fixture, &output.router_logits, layer_tolerance).await?;
    let checks = PromptChecks {
        maximum_layer_error,
        first_failing,
        routers_passed,
        first_router_mismatch,
    };
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
        Precision::F32 => verify_f32_decode(runner, fixture, input, tokens, checks).await,
        Precision::Bf16 => verify_bf16_decode(runner, input, tokens, checks).await,
    }
}

async fn compare_routers<B>(
    runner: &mut EngineRunner<B>,
    fixture: &golden_fixtures::PromptFixture,
    routers: &[EngineTensor],
    tolerance: f64,
) -> Result<(bool, Option<String>), Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let Some(expected) = fixture.router_logits() else {
        return Ok((routers.is_empty(), None));
    };
    let [layers, rows, experts]: [usize; 3] = expected
        .shape()
        .try_into()
        .map_err(|_| "router fixture must have rank three")?;
    if routers.len() != layers || experts < ROUTER_TOP_K {
        return Err("router output shape is incompatible with the fixture".into());
    }
    let layer_len = rows
        .checked_mul(experts)
        .ok_or("router fixture size overflowed")?;
    let mut passed = true;
    let mut first_mismatch = None;
    for (layer, router) in routers.iter().enumerate() {
        let actual = decode_f32_le(&runner.read(router).await?)?;
        let start = layer
            .checked_mul(layer_len)
            .ok_or("router fixture offset overflowed")?;
        let end = start
            .checked_add(layer_len)
            .ok_or("router fixture offset overflowed")?;
        let reference = expected
            .values()
            .get(start..end)
            .ok_or("router fixture is incomplete")?;
        let error = normwise_relative_error(reference, &actual)?;
        let layer_passed = error <= tolerance;
        if !layer_passed && first_mismatch.is_none() {
            first_mismatch = Some(format!("{}:*", layer + 1));
        }
        for (row, (reference, actual)) in reference
            .chunks_exact(experts)
            .zip(actual.chunks_exact(experts))
            .enumerate()
        {
            if !router_sets_match(reference, actual) && first_mismatch.is_none() {
                first_mismatch = Some(format!("{}:{row}", layer + 1));
            }
        }
        passed &= layer_passed;
        println!(
            "{}\trouter-{}\t{error:.8e}\t{}",
            fixture.name(),
            layer + 1,
            if layer_passed { "pass" } else { "FAIL" }
        );
    }
    passed &= first_mismatch.is_none();
    Ok((passed, first_mismatch))
}

fn router_sets_match(reference: &[f32], actual: &[f32]) -> bool {
    let expected = top_k_set(reference);
    let selected = top_k_set(actual);
    expected == selected || near_tie_swap(reference, &expected, &selected)
}

fn top_k_set(values: &[f32]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    indices.truncate(ROUTER_TOP_K);
    indices.sort_unstable();
    indices
}

fn near_tie_swap(reference: &[f32], expected: &[usize], actual: &[usize]) -> bool {
    let changed = expected
        .iter()
        .filter(|index| !actual.contains(index))
        .chain(actual.iter().filter(|index| !expected.contains(index)))
        .map(|&index| reference[index])
        .collect::<Vec<_>>();
    let minimum = changed.iter().copied().reduce(f32::min);
    let maximum = changed.iter().copied().reduce(f32::max);
    matches!((minimum, maximum), (Some(minimum), Some(maximum)) if maximum - minimum <= ROUTER_NEAR_TIE)
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
    prompt_tokens: Vec<u32>,
    checks: PromptChecks,
) -> Result<PromptVerification, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let PromptChecks {
        maximum_layer_error,
        first_failing,
        routers_passed,
        first_router_mismatch,
    } = checks;
    let prompt_kl =
        mean_logit_kl_divergence(fixture.prompt_logits().values(), &input.logits, input.vocab)?;
    let prompt = input.prompt;
    let expected_tokens = input.expected_tokens;
    let reference_logits = input.reference_logits;
    let prompt_length = input.prompt_tokens;
    let vocab = input.vocab;
    let steps = input.steps;
    let teacher_forced = decode(runner, input, DecodeMode::TeacherForced, true).await?;
    let output = runner
        .step(EngineStep {
            tokens: prompt_tokens,
            start_pos: 0,
            taps: false,
        })
        .await?
        .map_err(|error| format!("engine step failed: {error:?}"))?;
    let logits = decode_f32_le(&runner.read(&output.logits).await?)?;
    let free_running = decode(
        runner,
        DecodeInput {
            prompt,
            expected_tokens,
            reference_logits,
            prompt_tokens: prompt_length,
            logits,
            vocab,
            steps,
        },
        DecodeMode::FreeRunning,
        false,
    )
    .await?;
    let passed = first_failing.is_none()
        && routers_passed
        && prompt_kl <= LOGIT_KL_TOLERANCE
        && teacher_forced.mean_kl() <= LOGIT_KL_TOLERANCE
        && teacher_forced.agreement == teacher_forced.steps
        && free_running.mean_kl() <= LOGIT_KL_TOLERANCE
        && free_running.agreement == free_running.steps;
    println!(
        "{}\tmax-layer={maximum_layer_error:.8e}\tfirst-failing={}\tfirst-router-mismatch={}\tprompt-kl={prompt_kl:.8e}\tteacher-mean-kl={:.8e}\tteacher-max-kl={:.8e}\ttop-1={}/{}\tfree-top-1={}/{}\t{}",
        fixture.name(),
        layer_name(first_failing),
        first_router_mismatch.as_deref().unwrap_or("-"),
        teacher_forced.mean_kl(),
        teacher_forced.maximum_kl,
        teacher_forced.agreement,
        teacher_forced.steps,
        free_running.agreement,
        free_running.steps,
        if passed { "pass" } else { "FAIL" }
    );
    Ok(PromptVerification {
        maximum_layer_error,
        layers_passed: first_failing.is_none() && routers_passed,
        passed,
        teacher_forced,
        free_running,
    })
}

async fn verify_bf16_decode<B>(
    runner: &mut EngineRunner<B>,
    input: DecodeInput<'_>,
    prompt_tokens: Vec<u32>,
    checks: PromptChecks,
) -> Result<PromptVerification, Box<dyn Error>>
where
    B: Backend + Send + Sync + 'static,
{
    let PromptChecks {
        maximum_layer_error,
        first_failing,
        routers_passed,
        first_router_mismatch,
    } = checks;
    let prompt = input.prompt;
    let expected_tokens = input.expected_tokens;
    let reference_logits = input.reference_logits;
    let prompt_length = input.prompt_tokens;
    let vocab = input.vocab;
    let steps = input.steps;
    let teacher_forced = decode(runner, input, DecodeMode::TeacherForced, true).await?;
    let prompt_passed = first_failing.is_none()
        && routers_passed
        && teacher_forced.mean_kl() <= BF16_LOGIT_KL_TOLERANCE
        && top1_passes(teacher_forced);
    println!(
        "{}\tmax-layer={maximum_layer_error:.8e}\tfirst-failing={}\tfirst-router-mismatch={}\tteacher-mean-kl={:.8e}\tteacher-max-kl={:.8e}\ttop-1={}/{}\t{}",
        prompt,
        layer_name(first_failing),
        first_router_mismatch.as_deref().unwrap_or("-"),
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
        layers_passed: first_failing.is_none() && routers_passed,
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
    let weights = options.model_dir.join(fixtures.model_file());
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

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_bf16_olmoe_verification() -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let limits = forja_config::Limits {
            live_bytes: forja_config::ByteSize::new(48 * 1024 * 1024 * 1024),
            ..forja_config::Limits::default()
        };
        let options = Verify {
            engine: test_guests::olmoe().to_owned(),
            model_dir: root.join("OLMoE-1B-7B-0924"),
            fixtures: root.join("golden/OLMoE-1B-7B-0924"),
            backend: BackendArg::Metal,
            precision: Precision::Bf16,
            prompts: Vec::new(),
            graph_replay: forja_config::GraphReplay::Tier2,
            limits,
            scratch: root.join("scratch"),
        };
        tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_steps(&options, 32))
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_quantized_qwen3_coder_verification() -> Result<(), Box<dyn Error>> {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
        let limits = forja_config::Limits {
            live_bytes: forja_config::ByteSize::new(24 * 1024 * 1024 * 1024),
            ..forja_config::Limits::default()
        };
        let options = Verify {
            engine: test_guests::qwen3_coder().to_owned(),
            model_dir: root.join("Qwen3-Coder-30B-A3B-Instruct-4bit"),
            fixtures: root.join("golden/Qwen3-Coder-30B-A3B-Instruct-4bit"),
            backend: BackendArg::Metal,
            precision: Precision::F32,
            prompts: Vec::new(),
            graph_replay: forja_config::GraphReplay::Tier2,
            limits,
            scratch: root.join("scratch"),
        };
        tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(run_with_steps(&options, 8))
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
                    limits(&forja_config::Limits::default())?,
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
                "{{\"schema_version\":1,\"model\":{{\"file\":\"model.safetensors\",\"sha256\":\"{expected}\"}},\"prompts\":[]}}"
            ),
        )?;
        let options = Verify {
            engine: PathBuf::from("unused.wasm"),
            model_dir: model,
            fixtures: fixtures_path.clone(),
            backend: BackendArg::Cpu,
            precision: Precision::F32,
            prompts: Vec::new(),
            graph_replay: forja_config::GraphReplay::Tier2,
            limits: forja_config::Limits::default(),
            scratch: root.join("scratch"),
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
            graph_replay: forja_config::GraphReplay::Tier2,
            limits: forja_config::Limits::default(),
            scratch: root.join("scratch"),
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

    #[test]
    fn router_sets_allow_only_reference_near_ties() {
        let reference = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0005, 2.0];
        let mut near_tie = reference;
        near_tie[7] = 1.999;
        near_tie[8] = 2.001;
        assert!(router_sets_match(&reference, &near_tie));

        let mut mismatch = reference;
        mismatch[6] = 0.0;
        mismatch[8] = 3.5;
        assert!(!router_sets_match(&reference, &mismatch));
    }
}
