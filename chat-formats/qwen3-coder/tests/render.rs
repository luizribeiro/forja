//! Qwen3-Coder template compatibility checks.

use std::{env, fs, path::PathBuf};

use qwen3_coder_chat_format::{
    REPETITION_PENALTY, STOP_TOKEN_IDS, TEMPERATURE, TOP_K, TOP_P, render,
};

#[test]
fn matches_embedded_render_goldens() {
    let goldens: serde_json::Value = serde_json::from_str(include_str!(
        "../../qwen3-coder-xml/tests/fixtures/chat-template.json"
    ))
    .unwrap();
    for golden in goldens.as_array().unwrap() {
        let messages = serde_json::to_string(&golden["messages"]).unwrap();
        let actual = render(&messages, "null", r#"{"add_generation_prompt":true}"#).unwrap();
        assert_eq!(actual, golden["rendered"].as_str().unwrap());
    }
}

#[test]
fn disables_autoescape_and_renders_tools() {
    let plain = render(
        r#"[{"role":"user","content":"A & <B>"}]"#,
        "null",
        r#"{"add_generation_prompt":true}"#,
    )
    .unwrap();
    assert!(plain.contains("A & <B>"));

    let tools = r#"[{"type":"function","function":{"name":"lookup","description":"Look up <items> & values","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search & filter"}},"required":["query"]}}}]"#;
    let rendered = render(
        r#"[{"role":"user","content":"Find it"}]"#,
        tools,
        r#"{"add_generation_prompt":true}"#,
    )
    .unwrap();
    assert!(rendered.contains("<name>lookup</name>"));
    assert!(rendered.contains("Look up <items> & values"));

    let required = render(
        r#"[{"role":"user","content":"Find it"}]"#,
        tools,
        r#"{"add_generation_prompt":true,"tool_choice":"required"}"#,
    )
    .unwrap();
    assert!(required.ends_with("<|im_start|>assistant\n<tool_call>\n"));
}

#[test]
fn round_trips_openai_tool_calls_and_results() {
    let messages = r#"[{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"rust\"}"}}]},{"role":"tool","tool_call_id":"call_0","content":"found"}]"#;
    let rendered = render(messages, "null", r#"{"add_generation_prompt":true}"#).unwrap();
    assert!(rendered.contains("<parameter=query>\nrust\n</parameter>"));
    assert!(rendered.contains("<tool_response>\nfound\n</tool_response>"));
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn embedded_model_files_match_the_pinned_copy() {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").expect("FORJA_MODELS is not set"))
        .join("Qwen3-Coder-30B-A3B-Instruct-4bit");
    assert_eq!(
        fs::read_to_string(root.join("chat_template.jinja")).unwrap(),
        include_str!("../../qwen3-coder-xml/tests/fixtures/chat_template.jinja")
    );
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("generation_config.json")).unwrap()).unwrap();
    assert_eq!(config["eos_token_id"], serde_json::json!(STOP_TOKEN_IDS));
    assert_eq!(config["temperature"], TEMPERATURE);
    assert_eq!(config["top_p"], TOP_P);
    assert_eq!(config["top_k"], TOP_K);
    assert_eq!(config["repetition_penalty"], REPETITION_PENALTY);
}
