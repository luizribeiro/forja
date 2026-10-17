//! Qwen tool grammar and parser compatibility checks.

use forja_chat_format::{Event, ToolFormat};
use qwen3_coder_xml::{Options, Parser, Qwen3CoderXml};

#[test]
fn accepts_only_declared_quirk_options() {
    Qwen3CoderXml::options("{}").unwrap();
    assert!(Qwen3CoderXml::options(r#"{"unknown":true}"#).is_err());
    assert!(Qwen3CoderXml::options("null").is_err());
}

#[test]
fn grammar_selects_tools_and_preserves_parameter_order() {
    let tools = r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string"}},"required":["path","content","mode"]}}},{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}]"#;
    let all = Qwen3CoderXml::grammar(tools, "\"auto\"", "null", &Options).unwrap();
    assert_eq!(Qwen3CoderXml::trigger(&Options), "<tool_call>\n");
    assert!(all.contains("start: tool_0 | tool_1"));
    assert!(all.contains("tool_0: \"<function=read_file>\\n\" param_0_0 param_0_1 param_0_2"));
    let path = all.find("<parameter=path>").unwrap();
    let content = all.find("<parameter=content>").unwrap();
    let mode = all.find("<parameter=mode>").unwrap();
    assert!(path < content && content < mode);

    let named = Qwen3CoderXml::grammar(
        tools,
        r#"{"type":"function","function":{"name":"search"}}"#,
        r#"{"type":"text"}"#,
        &Options,
    )
    .unwrap();
    assert!(named.contains("start: tool_0"));
    assert!(!named.contains("read_file"));
}

#[test]
fn parses_hugging_face_tool_call_across_chunks() {
    let mut parser = Parser::default();
    let mut events = Vec::new();
    for chunk in "thinking<tool_|call>\n<function=get_weather>\n<para|meter=city>\nPar|is\n</parameter>\n</function>\n</tool_call>".split('|') {
        events.extend(parser.parse(chunk).unwrap());
    }
    assert_eq!(
        format!("{events:?}"),
        r#"[TextDelta("thinking"), CallStart("get_weather"), ArgumentDelta("city", "Paris"), CallEnd]"#
    );
    assert!(parser.parse("suffix").is_err());
}

#[test]
fn parser_emits_completed_parts_before_the_call_ends() {
    const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nMontréal\n</parameter>\n<parameter=unit>\n°C\n</parameter>\n</function>\n</tool_call>";
    let expected = Parser::default().parse(CALL).unwrap();
    let mut parser = Parser::default();
    let mut events = Vec::new();
    let mut pending = Vec::new();
    let mut arrivals = Vec::new();
    for (offset, byte) in CALL.bytes().enumerate() {
        pending.push(byte);
        match std::str::from_utf8(&pending) {
            Ok(chunk) => {
                let next = parser.parse(chunk).unwrap();
                arrivals.extend(std::iter::repeat_n(offset, next.len()));
                events.extend(next);
                pending.clear();
            }
            Err(error) if error.error_len().is_none() => {}
            Err(error) => panic!("invalid fixture UTF-8: {error}"),
        }
    }
    assert!(pending.is_empty());
    assert_eq!(events, expected);
    assert!(arrivals[0] < CALL.len() - CALL_END_LEN);
    assert!(arrivals[1] < CALL.len() - CALL_END_LEN);
    assert_eq!(arguments(&events), arguments(&expected));

    let char_boundaries = CALL
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(CALL.len()))
        .collect::<Vec<_>>();
    for seed in 1_u64..=32 {
        let mut state = seed;
        let mut start = 0;
        let mut split = Vec::new();
        while start + 1 < char_boundaries.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let remaining = char_boundaries.len() - start - 1;
            start += 1 + usize::try_from(state % u64::try_from(remaining).unwrap()).unwrap();
            split.push(char_boundaries[start]);
        }
        let mut parser = Parser::default();
        let mut actual = Vec::new();
        let mut begin = 0;
        for end in split {
            actual.extend(parser.parse(&CALL[begin..end]).unwrap());
            begin = end;
        }
        assert_eq!(actual, expected);
        assert_eq!(arguments(&actual), arguments(&expected));
    }
}

const CALL_END_LEN: usize = "</function>\n</tool_call>".len();

fn arguments(events: &[Event]) -> Vec<(&str, &str)> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::ArgumentDelta(name, value) => Some((name.as_str(), value.as_str())),
            _ => None,
        })
        .collect()
}

#[test]
fn enum_grammars_only_admit_declared_literals() {
    let tools = r#"[{"type":"function","function":{"name":"paint","parameters":{"type":"object","properties":{"color":{"type":"string","enum":["red","blue"]},"count":{"type":"integer","enum":[1,3]},"gloss":{"type":"number","enum":[0.5,1]},"dry":{"type":"boolean","enum":[true]}}}}}]"#;
    let grammar = Qwen3CoderXml::grammar(tools, "\"auto\"", "null", &Options).unwrap();
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
