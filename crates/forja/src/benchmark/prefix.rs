use super::{
    DecodeOptions, EngineDecode, EngineRunner, Error, Instant, PagedPrefix, Sample,
    measure_pipelined_tokens, prepare_decode, read_token, resident, timing,
};
#[derive(Clone)]
pub(super) struct Prepared {
    prefix: PagedPrefix,
    feedback: u32,
    tokens: u32,
}

impl Prepared {
    async fn new(
        runner: &mut EngineRunner<forja_metal::MetalBackend>,
        prompt: &[u32],
        options: DecodeOptions,
    ) -> Result<Self, Box<dyn Error>> {
        let tokens = u32::try_from(prompt.len())?;
        runner.wait_for_queued_submissions().await?;
        runner.prepare_paged_request(None, 0)?;
        if !runner.prepare_paged_request(None, u64::from(tokens))? {
            return Err("prefix workers require authenticated paged storage".into());
        }
        let feedback = prepare_decode(runner, prompt, options).await?;
        let prefix = runner.paged_prefix()?.ok_or("missing paged prefix")?;
        runner.prepare_paged_request(None, 0)?;
        Ok(Self {
            prefix,
            feedback,
            tokens,
        })
    }

    async fn seed(
        &self,
        runner: &mut EngineRunner<forja_metal::MetalBackend>,
        steps: usize,
        options: DecodeOptions,
    ) -> Result<forja_host::EngineDecodeOutput, Box<dyn Error>> {
        let reserved = u64::from(self.tokens)
            .checked_add(u64::try_from(steps)?)
            .and_then(|value| value.checked_add(1))
            .ok_or("decode prefix reservation overflowed")?;
        runner.wait_for_queued_submissions().await?;
        runner.prepare_paged_request(None, 0)?;
        let prefix = self
            .prefix
            .fork_reserved(u64::from(self.tokens), reserved)?;
        runner.prepare_paged_request(Some(prefix), reserved)?;
        let output = runner
            .decode(EngineDecode {
                tokens: Some(vec![self.feedback]),
                start_pos: self.tokens,
                sampling: options.sampling,
            })
            .await?
            .map_err(|error| format!("decode prefix seed failed: {error:?}"))?;
        read_token(&runner.read(&output.token).await?)?;
        runner.wait_for_queued_submissions().await?;
        Ok(output)
    }
}

pub(super) async fn measure(
    runner: &mut EngineRunner<forja_metal::MetalBackend>,
    prompt: &[u32],
    steps: usize,
    options: DecodeOptions,
    runners: &mut resident::Runners,
    prime: bool,
) -> Result<Sample, Box<dyn Error>> {
    let key = (resident::Phase::Single, prompt.len());
    let prepared = if let Some(prepared) = runners.prefixes.get(&key) {
        prepared.clone()
    } else {
        let started = Instant::now();
        let prepared = Prepared::new(runner, prompt, options).await?;
        runners.times.add(timing::Stage::Prefix, started.elapsed());
        runners.prefixes.insert(key, prepared.clone());
        prepared
    };
    let started = Instant::now();
    prepared.seed(runner, steps, options).await?;
    let setup = started.elapsed();
    let sample = measure_pipelined_tokens(runner, prepared.tokens, steps, options.sampling).await?;
    if prime {
        runners.times.add(timing::Stage::Prime, started.elapsed());
    } else {
        runners.times.add(timing::Stage::Setup, setup);
        runners.times.add(
            timing::Stage::Measured,
            std::time::Duration::from_secs_f64(sample.wall_seconds),
        );
    }
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn metal_prefix_forks_preserve_fresh_decode_logits() -> Result<(), Box<dyn Error>> {
        let models = std::env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?;
        let model = std::path::Path::new(&models).join("Qwen3-0.6B");
        let component = test_guests::qwen3();
        let crate::args::Command::Bench(options) = crate::args::parse([
            "bench".to_owned(),
            "--engine".to_owned(),
            component.display().to_string(),
            "--model".to_owned(),
            model.display().to_string(),
        ])?
        else {
            return Err("expected benchmark options".into());
        };
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(async {
                let (mut runner, info, _) =
                    super::super::load_benchmark_runner(&options, &options.points[0], component)
                        .await?;
                let decode = DecodeOptions {
                    host_argmax: false,
                    overlap: true,
                    sampling: forja_host::SamplingParams::default(),
                };
                for length in [7, 33, 65] {
                    let prompt = super::super::synthetic_tokens(length, info.vocab);
                    let cached = Prepared::new(&mut runner, &prompt, decode).await?;
                    let output = cached.seed(&mut runner, 8, decode).await?;
                    let actual = runner.read(&output.logits).await?;
                    super::super::prepare_benchmark_runner(&mut runner, length + 9)?;
                    prepare_decode(&mut runner, &prompt, decode).await?;
                    let output = runner
                        .decode(EngineDecode {
                            tokens: None,
                            start_pos: u32::try_from(length)?,
                            sampling: decode.sampling,
                        })
                        .await?
                        .map_err(|error| format!("fresh decode failed: {error:?}"))?;
                    assert_eq!(runner.read(&output.logits).await?, actual);
                    let output = cached.seed(&mut runner, 8, decode).await?;
                    assert_eq!(runner.read(&output.logits).await?, actual);
                }
                Ok::<_, Box<dyn Error>>(())
            })
    }
}
