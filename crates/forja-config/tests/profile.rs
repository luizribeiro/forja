//! Engine profile schema and validation checks.

use forja_config::{ActivationFormat, Profile, WeightFormat};

const PROFILE: &str = r#"
schema-version = 1
family = "qwen3"
default-tunings = ["residual-norm", "qk-norm-rope"]

[model]
id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
weights-sha256 = "sha256:f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"

[target]
component = "wasm32-wasip2"
backend = "metal"
device-family = "apple-m3-ultra"

[numerics]
activations = "bf16"
weights = "bf16"

[workload]
kind = "causal-lm"
batch = 1
max-context = 4096
prefill-chunk = 512

[variants.fixed]
"dense.decode" = "matmul.gemv-transposed"
"top-k" = "top-k.single-k8"

[variants.rules."attention.decode"]
parameter = "position"
arms = [
  { lo = 0, hi = 510, name = "sdpa.decomposed" },
  { lo = 511, hi = 1022, name = "sdpa.vector-single-pass" },
  { lo = 1023, hi = 4095, name = "sdpa.vector-two-pass" },
]
"#;

fn error(source: &str) -> String {
    toml::from_str::<Profile>(source)
        .unwrap_err()
        .message()
        .to_owned()
}

#[test]
fn parses_and_canonicalizes_a_dense_profile() {
    let profile: Profile = toml::from_str(PROFILE).unwrap();
    assert_eq!(profile.schema_version(), 1);
    assert_eq!(profile.family(), "qwen3");
    assert_eq!(profile.numerics().activations, ActivationFormat::Bf16);
    assert_eq!(profile.numerics().weights, WeightFormat::Bf16);
    assert_eq!(profile.default_tunings(), ["residual-norm", "qk-norm-rope"]);
    assert!(profile.quantization().is_none());

    let canonical = profile.canonical_toml().unwrap();
    let reparsed = toml::from_str::<Profile>(&canonical).unwrap();
    assert_eq!(reparsed, profile);
    assert_eq!(canonical, profile.canonical_toml().unwrap());
    assert_eq!(profile.sha256().unwrap().len(), 71);
    assert_eq!(profile.sha256().unwrap(), reparsed.sha256().unwrap());
}

#[test]
fn rejects_unknown_fields_and_invalid_identity() {
    assert!(error(&PROFILE.replace("family =", "families =")).contains("unknown field `families`"));
    assert!(
        error(&PROFILE.replace(
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "C1899de289a04d12100db370d81485cdf75e47ca"
        ))
        .contains("40 lowercase hexadecimal digits")
    );
    assert!(
        error(&PROFILE.replace("sha256:f47f", "sha256:F47f"))
            .contains("64 lowercase hexadecimal digits")
    );
    assert!(error(&PROFILE.replace("sha256:f47f", "f47f")).contains("must start with \"sha256:\""));
    for id in ["..", "a/..", "/abs", "a//b"] {
        assert!(
            error(&PROFILE.replace("Qwen/Qwen3-0.6B", id)).contains("relative segments"),
            "{id}"
        );
    }
}

#[test]
fn rejects_invalid_workload_bounds_and_tuning_names() {
    for source in [
        PROFILE.replace("batch = 1", "batch = 0"),
        PROFILE.replace("max-context = 4096", "max-context = 0"),
        PROFILE.replace("prefill-chunk = 512", "prefill-chunk = 0"),
    ] {
        assert!(toml::from_str::<Profile>(&source).is_err(), "{source}");
    }
    assert!(
        error(&PROFILE.replace("prefill-chunk = 512", "prefill-chunk = 4097"))
            .contains("must not exceed")
    );
    assert!(
        error(&PROFILE.replace(
            "[\"residual-norm\", \"qk-norm-rope\"]",
            "[\"residual-norm\", \"residual-norm\"]"
        ))
        .contains("must be unique")
    );
}

#[test]
fn rejects_duplicate_sites_and_incomplete_or_overlapping_rules() {
    let duplicate = PROFILE.replace(
        "\"dense.decode\" = \"matmul.gemv-transposed\"",
        "\"attention.decode\" = \"sdpa.decomposed\"",
    );
    assert!(error(&duplicate).contains("appears in fixed and rules"));

    for source in [
        PROFILE.replace("lo = 511", "lo = 512"),
        PROFILE.replace("lo = 511", "lo = 510"),
        PROFILE.replace("hi = 4095", "hi = 4094"),
        PROFILE.replace("lo = 1023, hi = 4095", "lo = 1023, hi = 1022"),
    ] {
        let message = error(&source);
        assert!(
            message.contains("ordered, complete, and non-overlapping")
                || message.contains("cover 0 through 4095"),
            "{message}"
        );
    }
}

#[test]
fn accepts_quantized_profiles_and_rejects_quantization_for_dense_weights() {
    let quantization = r#"

[quantization]
scheme = "mlx-affine"
default-bits = 4
group-size = 64

[quantization.overrides."model.layers.*.mlp.gate.weight"]
bits = 8
"#;
    let dense = format!("{PROFILE}{quantization}");
    assert!(error(&dense).contains("dense profiles must not contain"));

    let q4 = dense.replace("weights = \"bf16\"", "weights = \"q4\"");
    let profile: Profile = toml::from_str(&q4).unwrap();
    assert_eq!(profile.numerics().weights, WeightFormat::Q4);
    assert_eq!(profile.quantization().unwrap().default_bits, 4);

    let missing = PROFILE.replace("weights = \"bf16\"", "weights = \"q4\"");
    assert!(error(&missing).contains("q4 weights require"));
}

#[test]
fn enforces_profile_collection_and_name_bounds() {
    fn strings(count: usize, prefix: &str) -> String {
        (0..count)
            .map(|index| format!("\"{prefix}{index}\""))
            .collect::<Vec<_>>()
            .join(", ")
    }

    let at_tuning_limit = PROFILE.replace(
        "[\"residual-norm\", \"qk-norm-rope\"]",
        &format!("[{}]", strings(64, "t")),
    );
    assert!(toml::from_str::<Profile>(&at_tuning_limit).is_ok());
    assert!(
        error(&at_tuning_limit.replace(&strings(64, "t"), &strings(65, "t"))).contains("64-entry")
    );

    let fixed = |count| {
        (0..count)
            .map(|index| format!("site-{index} = \"variant\""))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let fixed_profile = PROFILE.replace(
        "\"dense.decode\" = \"matmul.gemv-transposed\"\n\"top-k\" = \"top-k.single-k8\"",
        &fixed(1_024),
    );
    assert!(toml::from_str::<Profile>(&fixed_profile).is_ok());
    assert!(error(&fixed_profile.replace(&fixed(1_024), &fixed(1_025))).contains("1024-entry"));

    let rules = |count| {
        (0..count)
            .map(|index| {
                format!(
                    "[variants.rules.site-{index}]\nparameter = \"position\"\narms = [{{ lo = 0, hi = 4095, name = \"variant\" }}]"
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let rules_profile = PROFILE
        .split("[variants.rules.\"attention.decode\"]")
        .next()
        .unwrap()
        .to_owned()
        + &rules(1_024);
    assert!(toml::from_str::<Profile>(&rules_profile).is_ok());
    assert!(
        error(
            &(rules_profile[..rules_profile.len() - rules(1_024).len()].to_owned() + &rules(1_025))
        )
        .contains("1024-entry")
    );

    let tuning_variants = |count| {
        (0..count)
            .map(|index| format!("[tuning-variants.t{index}]"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let variants_profile = format!("{PROFILE}\n{}", tuning_variants(64));
    assert!(toml::from_str::<Profile>(&variants_profile).is_ok());
    assert!(error(&format!("{PROFILE}\n{}", tuning_variants(65))).contains("64-entry"));

    let name = "n".repeat(64);
    assert!(toml::from_str::<Profile>(&PROFILE.replace("residual-norm", &name)).is_ok());
    assert!(error(&PROFILE.replace("residual-norm", &(name + "n"))).contains("64-byte"));
}
