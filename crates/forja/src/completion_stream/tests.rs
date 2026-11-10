use super::*;
use serde_json::json;

pub(crate) fn pair() -> (Value, Value) {
    let plain = json!({"kind":"completion-benchmark", "schema_version":1,"mode":"plain",
        "provenance":{"commit":"fixture","dirty":false,"clean_source":true,"binary_sha256":"a".repeat(64),
            "device":"CPU","machine":"fixture","architecture":"test","os":"test","os_version":"1","os_build":"1",
            "build_environment":"fixture","gpu_cores":null,"memory_bytes":null},
        "stream_config":{"inputs":{"engine_sha256":"engine","profile_hash":"profile","weights_sha256":"weights"},"sampling":"greedy"},
        "measurement_mode":{"execution":"scheduled-request-groups"},
        "results":[{"identity":{"name":"control","seed":42,"prompt_sha256":format!("sha256:{}", "c".repeat(64)),
            "prompt_tokens":4097,"output_tokens":256,"concurrency":1,"reps":1},"rounds":0,"minimum_speedup":0.98,
            "token_stream_hashes":[format!("sha256:{}", "b".repeat(64))],
            "samples":[{"repetition":0,"wall_seconds":1.0,"decode_wall_seconds":1.0,"gpu_seconds":0.0,
                "plain_gpu_seconds":0.0,"verify_gpu_seconds":0.0,"prompt_tokens":[4097],"output_tokens":[256],"finish_reasons":["length"]}]}]});
    let mut draft = plain.clone();
    draft["mode"] = "draft".into();
    (plain, draft)
}

#[test]
fn refuses_mismatches_and_missing_exactness_evidence() {
    let (plain, draft) = pair();
    assert_matches(&plain, &draft, Comparison::Candidate).unwrap();
    for pointer in [
        "/mode",
        "/stream_config/inputs/engine_sha256",
        "/stream_config/sampling",
        "/results/0/identity/seed",
        "/results/0/identity/prompt_sha256",
        "/results/0/identity/reps",
        "/results/0/identity/concurrency",
        "/results/0/token_stream_hashes/0",
    ] {
        let mut changed = draft.clone();
        *changed.pointer_mut(pointer).unwrap() = Value::Null;
        assert!(
            assert_matches(&plain, &changed, Comparison::Candidate).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "/results/0/identity/seed",
        "/results/0/identity/prompt_sha256",
        "/results/0/identity/reps",
        "/results/0/identity/concurrency",
    ] {
        let mut incomplete = plain.clone();
        *incomplete.pointer_mut(pointer).unwrap() = Value::Null;
        let mut other = incomplete.clone();
        other["mode"] = "draft".into();
        assert!(assert_matches(&incomplete, &other, Comparison::Candidate).is_err());
    }
}

#[test]
fn provenance_comparison_requires_explicit_baseline_labelling() {
    let (plain, mut draft) = pair();
    draft["provenance"]["binary_sha256"] = json!("d".repeat(64));
    draft["provenance"]["commit"] = "candidate".into();
    draft["stream_config"]["inputs"]["engine_sha256"] = "candidate-engine".into();
    draft["stream_config"]["inputs"]["profile_hash"] = "candidate-profile".into();
    assert!(assert_matches(&plain, &draft, Comparison::Candidate).is_err());
    assert_matches(&plain, &draft, Comparison::BaselineVsCandidate).unwrap();
    for (pointer, replacement) in [
        ("/provenance/dirty", json!(true)),
        ("/provenance/clean_source", json!(false)),
        ("/provenance/device", json!("other")),
        ("/stream_config/inputs/weights_sha256", json!("other")),
    ] {
        let mut changed = draft.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            assert_matches(&plain, &changed, Comparison::BaselineVsCandidate).is_err(),
            "{pointer}"
        );
        let mut changed_plain = plain.clone();
        changed_plain["provenance"]["dirty"] = true.into();
        assert!(assert_matches(&changed_plain, &draft, Comparison::BaselineVsCandidate).is_err());
    }
}

#[test]
fn command_reads_records_and_propagates_validation_and_io_errors() {
    let root = forja_testing::temporary_directory("stream-command").unwrap();
    let (plain, mut draft) = pair();
    let mut options = Options {
        plain: root.join("plain.json"),
        draft: root.join("draft.json"),
        comparison: Comparison::Candidate,
    };
    fs::write(&options.plain, serde_json::to_vec(&plain).unwrap()).unwrap();
    fs::write(&options.draft, serde_json::to_vec(&draft).unwrap()).unwrap();
    run(&options).unwrap();
    draft["provenance"]["binary_sha256"] = json!("d".repeat(64));
    fs::write(&options.draft, serde_json::to_vec(&draft).unwrap()).unwrap();
    assert!(run(&options).is_err());
    options.comparison = Comparison::BaselineVsCandidate;
    run(&options).unwrap();
    fs::write(&options.draft, b"invalid JSON").unwrap();
    assert!(run(&options).is_err());
    fs::remove_file(&options.draft).unwrap();
    assert!(run(&options).is_err());
    fs::remove_dir_all(root).unwrap();
}
