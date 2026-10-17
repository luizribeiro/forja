//! Qwen tool grammar and parser compatibility checks.

use qwen3_coder_chat_format::grammar;

#[test]
fn grammar_selects_tools_and_preserves_parameter_order() {
    let tools = r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer"}},"required":["path"]}}},{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]"#;
    let all = grammar(tools, "\"auto\"", "null").unwrap();
    assert_eq!(all.1, "<tool_call>\n");
    assert!(all.0.contains("start: tool_0 | tool_1"));
    assert!(all.0.contains("[param_0_0] param_0_1"));

    let named = grammar(
        tools,
        r#"{"type":"function","function":{"name":"search"}}"#,
        r#"{"type":"text"}"#,
    )
    .unwrap();
    assert!(named.0.contains("start: tool_0"));
    assert!(!named.0.contains("read_file"));
}

#[test]
fn enum_grammars_only_admit_declared_literals() {
    let tools = r#"[{"type":"function","function":{"name":"paint","parameters":{"type":"object","properties":{"color":{"type":"string","enum":["red","blue"]},"count":{"type":"integer","enum":[1,3]},"gloss":{"type":"number","enum":[0.5,1]},"dry":{"type":"boolean","enum":[true]}}}}}]"#;
    let (grammar, _) = grammar(tools, "\"auto\"", "null").unwrap();
    for rule in [
        r#""red" | "blue""#,
        r#""1" | "3""#,
        r#""0.5" | "1""#,
        r#""true""#,
    ] {
        assert!(grammar.contains(rule), "missing {rule} in {grammar}");
    }
    assert!(
        !grammar
            .lines()
            .any(|line| line.starts_with("param_") && line.contains(" STRING "))
    );
}
