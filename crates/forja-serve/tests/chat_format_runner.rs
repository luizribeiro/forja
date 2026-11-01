//! Sandboxing checks for chat-format components.

use forja_serve::{ChatEvent, ChatFormat, ChatFormatLimits};

#[test]
fn runs_a_bounded_zero_import_chat_format() {
    let limits = ChatFormatLimits {
        fuel: 10_000_000,
        input_bytes: 4096,
        output_bytes: 4096,
        ..ChatFormatLimits::default()
    };
    let mut format = ChatFormat::load(test_guests::chat_format_smoke(), limits).unwrap();
    let info = format.info();
    assert_eq!(info.stop_token_ids, [2]);
    assert_eq!(
        format.render("[]", "[]", "{}").unwrap().unwrap(),
        "rendered:[]"
    );
    let output_error = format.render("[]", "[]", r#"{"large":true}"#).unwrap_err();
    assert_eq!(
        output_error.to_string(),
        "chat format output exceeds its byte limit"
    );
    let input_error = ChatFormat::load(
        test_guests::chat_format_smoke(),
        ChatFormatLimits {
            input_bytes: 5,
            ..ChatFormatLimits::default()
        },
    )
    .unwrap()
    .render("[]", "[]", "{}")
    .unwrap_err();
    assert_eq!(
        input_error.to_string(),
        "chat format input exceeds its byte limit"
    );

    let clock: serde_json::Value = serde_json::from_str(
        &format
            .render_at("[]", "[]", r#"{"clock":true}"#, 1_798_761_600, 3600)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(clock["now_unix_seconds"], 1_798_761_600);
    assert_eq!(clock["now_utc_offset_seconds"], 3600);

    format.grammar("[]", "null", "null").unwrap().unwrap();
    let events = format.parse(7, "hello").unwrap().unwrap();
    assert!(matches!(&events[..], [ChatEvent::TextDelta(value)] if value == "hello"));
    format.reset(7).unwrap();
    assert!(format.render("[]", "[]", r#"{"loop":true}"#).is_err());

    let limits = ChatFormatLimits {
        trigger_bytes: 1,
        ..ChatFormatLimits::default()
    };
    let mut limited = ChatFormat::load(test_guests::chat_format_smoke(), limits).unwrap();
    assert!(limited.grammar("[]", "null", "null").is_err());
}

#[test]
fn component_declares_no_imports() {
    let engine = wasmtime::Engine::default();
    for path in [
        test_guests::chat_format_smoke(),
        test_guests::qwen3_coder_chat_format(),
        test_guests::custom_chat_format(),
        test_guests::qwen38_chat_format(),
    ] {
        let component = wasmtime::component::Component::from_file(&engine, path).unwrap();
        assert_eq!(
            component.component_type().imports(&engine).count(),
            0,
            "{}",
            path.display()
        );
    }
}

#[test]
fn runs_a_custom_chat_format_component() {
    let mut format = ChatFormat::load(
        test_guests::custom_chat_format(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    assert_eq!(format.info().stop_token_ids, [7]);
    assert_eq!(
        format
            .render(r#"[{"content":"custom"}]"#, "null", "{}")
            .unwrap()
            .unwrap(),
        "custom"
    );
}

#[test]
fn runs_a_profile_chat_format() {
    let mut format = ChatFormat::load(
        test_guests::qwen3_coder_chat_format(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    let info = format.info();
    assert_eq!(info.stop_token_ids, [151_645, 151_643]);
    assert_eq!(info.sampling.top_k, Some(20));
    assert_eq!(format.load_warnings().len(), 1);
    assert!(format.load_warnings()[0].contains("will not be applied"));
    let rendered = format
        .render(
            r#"[{"role":"user","content":"A & <B>"}]"#,
            "null",
            r#"{"add_generation_prompt":true}"#,
        )
        .unwrap()
        .unwrap();
    assert!(rendered.contains("A & <B>"));
    let tools =
        r#"[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]"#;
    let grammar = format.grammar(tools, "\"auto\"", "null").unwrap().unwrap();
    assert_eq!(grammar.trigger, "<tool_call>\n");
    let started = format
        .parse(9, "<tool_call>\n<function=lookup>\n")
        .unwrap()
        .unwrap();
    assert!(matches!(&started[..], [ChatEvent::CallStart(name)] if name == "lookup"));
    let events = format
        .parse(9, "</function>\n</tool_call>")
        .unwrap()
        .unwrap();
    assert!(matches!(&events[..], [ChatEvent::CallEnd]));
    format.reset(9).unwrap();
    let reset = format.parse(9, "plain").unwrap().unwrap();
    assert!(matches!(&reset[..], [ChatEvent::TextDelta(text)] if text == "plain"));
}

#[test]
fn reuses_a_profile_component_for_multibyte_tool_conversations() {
    let mut format = ChatFormat::load(
        test_guests::qwen3_coder_chat_format(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    let tools = r#"[{"type":"function","function":{"name":"lookup","description":"Busca café em 東京 🧰","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]"#;
    let calls = [
        serde_json::json!([{"role": "user", "content": "café 東京 🧰"}]),
        serde_json::json!([
            {"role": "user", "content": "café 東京 🧰"},
            {"role": "assistant", "content": "déjà vu"}
        ]),
        serde_json::json!([
            {"role": "user", "content": "café 東京 🧰"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{\"city\":\"Montréal\"}"}
            }]},
            {"role": "tool", "content": "résultat 東京 🧰", "tool_call_id": "call_0"}
        ]),
    ];
    for iteration in 0..2_000_u64 {
        let messages =
            serde_json::to_string(&calls[usize::try_from(iteration % 3).unwrap()]).unwrap();
        let rendered = format
            .render(&messages, tools, r#"{"add_generation_prompt":true}"#)
            .unwrap_or_else(|error| panic!("render {iteration} trapped: {error}"))
            .unwrap_or_else(|error| panic!("render {iteration} failed: {error}"));
        assert!(rendered.contains("café 東京 🧰"), "render {iteration}");

        let stream = iteration % 4;
        let mut events = format
            .parse(stream, "préface 東京 🧰<tool_")
            .unwrap()
            .unwrap();
        events.extend(
            format
                .parse(
                    stream,
                    "call>\n<function=lookup>\n<parameter=city>\nMontréal",
                )
                .unwrap()
                .unwrap(),
        );
        events.extend(
            format
                .parse(stream, " 東京 🧰\n</parameter>\n</function>\n</tool_call>")
                .unwrap()
                .unwrap(),
        );
        assert!(matches!(
            &events[..],
            [
                ChatEvent::TextDelta(text),
                ChatEvent::CallStart(name),
                ChatEvent::ArgumentDelta((parameter, value)),
                ChatEvent::CallEnd
            ] if text == "préface 東京 🧰"
                && name == "lookup"
                && parameter == "city"
                && value == "Montréal 東京 🧰"
        ));
        format.reset(stream).unwrap();
    }
}

#[test]
fn grammar_receives_bounded_opaque_request_options() {
    let options = r#"{"echo_options":true,"parallel_tool_calls":false}"#;
    let mut format = ChatFormat::load(
        test_guests::chat_format_smoke(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    assert_eq!(
        format
            .grammar_with_options("[]", "null", "null", options)
            .unwrap()
            .unwrap()
            .grammar,
        options
    );
    let mut limited = ChatFormat::load(
        test_guests::chat_format_smoke(),
        ChatFormatLimits {
            input_bytes: options.len(),
            ..ChatFormatLimits::default()
        },
    )
    .unwrap();
    assert!(
        limited
            .grammar_with_options("[]", "null", "null", options)
            .is_err()
    );
}

#[tokio::test]
async fn repeated_calls_keep_the_grammar_open_until_eos() {
    use forja_serve::guidance::GuidanceWorker;
    let body = "<function=lookup>\n</function>\n</tool_call>";
    let separator = "\n<tool_call>\n";
    let vocab = [
        (body.replace('\n', "Ċ"), 0),
        (separator.replace('\n', "Ċ"), 1),
        ("<eos>".to_owned(), 2),
    ];
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, Vec::new())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let worker = GuidanceWorker::new(&tokenizer, 3, &[2]).unwrap();
    let mut format = ChatFormat::load(
        test_guests::qwen38_chat_format(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    let tools =
        r#"[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]"#;
    let grammar = format
        .grammar(tools, "\"required\"", "null")
        .unwrap()
        .unwrap();
    let session = worker.compile(grammar.grammar).await.unwrap();
    assert_eq!(session.mask().await.unwrap().unwrap(), [1]);
    session.commit(0).await.unwrap();
    assert_eq!(session.mask().await.unwrap().unwrap(), [6]);
    session.commit(1).await.unwrap();
    assert_eq!(session.mask().await.unwrap().unwrap(), [1]);
    session.commit(0).await.unwrap();
    assert_eq!(session.mask().await.unwrap().unwrap(), [6]);
    session.commit(2).await.unwrap();
    assert!(session.mask().await.unwrap().is_none());
    let events = format
        .parse(
            1,
            &format!("{}{}{}{}", grammar.trigger, body, separator, body),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ChatEvent::CallEnd))
            .count(),
        2
    );
    format.reset(1).unwrap();
    let grammar = format
        .grammar_with_options(
            tools,
            "\"required\"",
            "null",
            r#"{"parallel_tool_calls":false}"#,
        )
        .unwrap()
        .unwrap();
    let single = worker.compile(grammar.grammar).await.unwrap();
    assert_eq!(single.mask().await.unwrap().unwrap(), [1]);
    single.commit(0).await.unwrap();
    assert!(single.mask().await.unwrap().is_none());
}

#[test]
fn pinned_chat_component_matches_transformers_render_goldens() {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../../chat-formats/qwen38/tests/fixtures/render.json"
    ))
    .unwrap();
    let mut format = ChatFormat::load(
        test_guests::qwen38_chat_format(),
        ChatFormatLimits::default(),
    )
    .unwrap();
    assert_eq!(format.info().stop_token_ids, [248_046, 248_044]);
    for fixture in fixtures.as_array().unwrap() {
        let options = serde_json::json!({"add_generation_prompt":fixture["options"]["add_generation_prompt"], "chat_template_kwargs": fixture["options"].as_object().unwrap().iter().filter(|(name,_)| *name != "add_generation_prompt").map(|(key,value)| (key.clone(),value.clone())).collect::<serde_json::Map<_,_>>()});
        let rendered = format
            .render(
                &fixture["messages"].to_string(),
                &fixture["tools"].to_string(),
                &options.to_string(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(rendered, fixture["rendered"], "{}", fixture["name"]);
    }
}
