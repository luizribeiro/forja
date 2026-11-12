use super::{BatchShape, EngineRunner, Error, PagedPrefix, Profile, run_serving_profile_step};
use forja_host::{CanonicalPrompt, HybridPrefix, SnapshotAllowance};
use std::sync::Arc;

type Runner = EngineRunner<forja_metal::MetalBackend>;

pub(super) enum Prefix {
    Paged(PagedPrefix),
    Paired(
        Arc<CanonicalPrompt>,
        Arc<HybridPrefix<forja_metal::MetalBackend>>,
    ),
}

impl Prefix {
    pub(super) async fn prepare(
        runner: &mut Runner,
        tokens: &[u32],
        options: &Profile,
        shape: BatchShape,
        reserved: u64,
    ) -> Result<Self, Box<dyn Error>> {
        runner.prepare_paged_lane(0, None, 0)?;
        let paired = if runner.supports_prefix_reuse() {
            runner.prepare_paged_lane(0, None, reserved)?;
            None
        } else {
            let profile = crate::resolution::read_embedded_profile(&options.engine)?;
            let declared = profile
                .serving()
                .and_then(|serving| serving.state_snapshot_bytes)
                .ok_or("paired profiling requires a declared snapshot allowance")?
                .get();
            let classes = profile
                .variants()
                .fixed
                .values()
                .map(String::as_str)
                .collect::<Vec<_>>();
            // This prefix is private to this runner and never enters a persistent cache.
            let namespace = runner.canonical_prefix_namespace([0; 32], &classes)?;
            let prompt = runner.canonical_prompt(namespace, tokens.to_vec())?;
            runner.prepare_canonical_lane(0, prompt.clone(), None, reserved)?;
            Some((prompt, SnapshotAllowance::new(declared, None)?))
        };
        let pieces = super::prompt_submissions(
            &tokens[..options.prefix_context],
            0,
            std::num::NonZeroUsize::new(usize::try_from(shape.tokens)?)
                .ok_or("empty prompt shape")?,
            |start, count| runner.prompt_piece_limit(start, count).map_err(Into::into),
        )?;
        for (start, chunk) in pieces {
            run_serving_profile_step(runner, shape, chunk, usize::try_from(start)?).await?;
        }
        let prefix = if let Some((prompt, allowance)) = paired {
            Self::Paired(prompt, runner.capture_canonical_prefix(0, allowance)?)
        } else {
            Self::Paged(
                runner
                    .paged_prefix()?
                    .ok_or("serving prefill did not produce a paged prefix")?,
            )
        };
        runner.prepare_paged_lane(0, None, 0)?;
        Ok(prefix)
    }

    pub(super) fn restore(
        &self,
        runner: &mut Runner,
        tokens: u64,
        reserved: u64,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Paged(prefix) => {
                runner.prepare_paged_lane(
                    0,
                    Some(prefix.fork_reserved(tokens, reserved)?),
                    reserved,
                )?;
            }
            Self::Paired(prompt, prefix) => {
                if runner.prepare_canonical_lane(0, prompt.clone(), Some(prefix), reserved)?
                    != tokens
                {
                    return Err("paired profile prefix differs from its declared extent".into());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_suffix_profile_restores_paired_carry_and_attention() -> Result<(), Box<dyn Error>> {
        let models = std::env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?;
        let model = std::path::Path::new(&models).join("Qwen3.8-27B-4bit");
        let crate::args::Command::Profile(mut options) = crate::args::parse([
            "--isolated".to_owned(),
            "profile".to_owned(),
            "--engine".to_owned(),
            test_guests::qwen38().display().to_string(),
            "--model".to_owned(),
            model.display().to_string(),
            "--mode".to_owned(),
            "prefill".to_owned(),
            "--context".to_owned(),
            "65".to_owned(),
            "--prefix-context".to_owned(),
            "64".to_owned(),
            "--step-tokens".to_owned(),
            "64".to_owned(),
        ])?
        else {
            return Err("expected profile options".into());
        };
        options.warmups = 0;
        let measured = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(super::super::measure_token_profile(&options))?;
        assert_eq!(measured.step.submissions.len(), 1);
        assert!(measured.cache_setup_wall > std::time::Duration::ZERO);
        assert!(measured.baseline_gpu > std::time::Duration::ZERO);
        Ok(())
    }
    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_native_profile_keeps_every_context_bounded_piece() -> Result<(), Box<dyn Error>> {
        let models = std::env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?;
        let crate::args::Command::Profile(mut options) = crate::args::parse([
            "--isolated".to_owned(),
            "profile".to_owned(),
            "--engine".to_owned(),
            test_guests::qwen38().display().to_string(),
            "--model".to_owned(),
            std::path::Path::new(&models)
                .join("Qwen3.8-27B-4bit")
                .display()
                .to_string(),
            "--mode".to_owned(),
            "prefill".to_owned(),
            "--context".to_owned(),
            "513".to_owned(),
        ])?
        else {
            return Err("expected profile options".into());
        };
        options.warmups = 0;
        let measured = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(super::super::measure_token_profile(&options))?;
        assert!(measured.step.submissions.len() >= 2);
        for submission in &measured.step.submissions {
            assert!(submission.predicted_gpu <= std::time::Duration::from_secs(8));
            assert_ne!(submission.buffer_admission.len(), 0);
            assert!(submission.buffer_admission.iter().all(|buffer| {
                buffer.backstop_ratio == 4
                    && buffer.predicted_gpu <= std::time::Duration::from_secs(2)
            }));
        }
        assert_eq!(
            measured
                .step
                .submissions
                .iter()
                .map(|s| s.gpu_time)
                .sum::<std::time::Duration>(),
            measured.profiled_gpu
        );
        Ok(())
    }
}
