use super::*;

#[tokio::test]
async fn metal_classic_paged_repetitions_start_fresh_requests() -> Result<(), Box<dyn Error>> {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../forja-host/tests/fixtures/qwen38");
    let crate::args::Command::Bench(options) = crate::args::parse([
        "--isolated".to_owned(),
        "bench".to_owned(),
        "--engine".to_owned(),
        test_guests::qwen38_tiny_metal().display().to_string(),
        "--model".to_owned(),
        model.display().to_string(),
        "--pp=7".to_owned(),
        "--tg=2".to_owned(),
        "--reps=2".to_owned(),
        "--set=bench.decode_prefill=3".to_owned(),
        "--set=bench.contexts=[4]".to_owned(),
        "--set=bench.warmups=2".to_owned(),
    ])?
    else {
        return Err("expected benchmark options".into());
    };
    let mut runner = EngineRunner::new(
        test_guests::qwen38_tiny_metal(),
        forja_metal::MetalBackend::new()?,
        crate::engine::limits_for_live_bytes(&options.limits, 1 << 30)?,
        model.join("model.safetensors"),
    )
    .await?;
    runner
        .load_with_selections(
            Some(1),
            &forja_host::EngineLoadConfig {
                max_context: Some(64),
                paged_cache: true,
                kv_cache: forja_host::KvCacheFormat::F16,
                fixed_variant_picks: [
                    ("quant", "quant-matmul.portable"),
                    ("quant-decode", "quant-matmul.portable"),
                    ("quant-decode-batch", "quant-matmul.portable"),
                    ("quant-decode-batch-wide", "quant-matmul.portable"),
                    ("attention", "sdpa.decomposed"),
                    ("sample", "sample.parallel"),
                    ("delta-prefill", "delta-rule-block.wy64-fixed"),
                    ("paged-attention.prefill", "paged-sdpa.decomposed"),
                    ("paged-attention.decode", "paged-sdpa.decomposed"),
                ]
                .into_iter()
                .map(|(site, name)| forja_host::FixedVariantPick {
                    site: site.to_owned(),
                    name: name.to_owned(),
                })
                .collect(),
                ..Default::default()
            },
        )
        .await??;
    let point = &options.points[0];
    let tokens = synthetic_tokens(point.pp, 33);
    let mut runners = resident::Runners::new(false);
    prepare_benchmark_runner(&mut runner, point.pp)?;
    for (host_argmax, overlap) in [(true, false), (false, false), (false, true)] {
        let decode = decode_options(point, host_argmax, overlap);
        prime_benchmark(&mut runner, &tokens, point, &options, decode, &mut runners).await?;
        for _ in 0..options.reps {
            measure_record_prefill(&mut runner, &tokens, &mut runners, false).await?;
            measure_record_decode(
                &mut runner,
                &tokens[..options.decode_prefill],
                point.tg,
                decode,
                &mut runners,
                false,
            )
            .await?;
        }
        let (digest, _) = benchmark_output(
            &mut runner,
            &tokens[..options.decode_prefill],
            point.tg,
            decode.sampling,
            &runners,
        )
        .await?;
        assert_ne!(digest, "");
    }
    Ok(())
}
