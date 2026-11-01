use serde::{Deserialize, Serialize};

/// Text-only chat content in either supported `OpenAI` representation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// One UTF-8 string.
    Text(String),
    /// An ordered list of text parts.
    Parts(Vec<TextPart>),
}

/// One explicitly tagged text part.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TextPart {
    /// UTF-8 text without media capabilities.
    Text {
        /// Content appended in order.
        text: String,
    },
}

impl From<String> for MessageContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for MessageContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl MessageContent {
    pub(crate) fn chunks(&self) -> impl Iterator<Item = &str> {
        let (text, parts): (Option<&str>, &[TextPart]) = match self {
            Self::Text(text) => (Some(text), &[]),
            Self::Parts(parts) => (None, parts),
        };
        text.into_iter()
            .chain(parts.iter().map(|TextPart::Text { text }| text.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_parts_round_trip_without_accepting_media() {
        let source = r#"[{"type":"text","text":"Hello "},{"type":"text","text":"Montréal"}]"#;
        let content: MessageContent = serde_json::from_str(source).unwrap();
        assert_eq!(content.chunks().collect::<String>(), "Hello Montréal");
        assert_eq!(serde_json::to_string(&content).unwrap(), source);
        for source in [
            r#"[{"type":"image_url","image_url":"x"}]"#,
            r#"[{"type":"video","video":"x"}]"#,
            r#"[{"type":"text","text":"x","image":"x"}]"#,
        ] {
            assert!(serde_json::from_str::<MessageContent>(source).is_err());
        }
    }
}
