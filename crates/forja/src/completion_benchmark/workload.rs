use std::{collections::BTreeSet, error::Error, fs, path::Path};

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Suite {
    pub schema_version: u32,
    pub source: String,
    pub target: String,
    pub padding: String,
    #[serde(default)]
    pub padding_prefix: String,
    #[serde(default)]
    pub padding_suffix: String,
    pub context_semantics: String,
    pub policy: serde_json::Value,
    pub rows: Vec<Row>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Row {
    pub name: String,
    pub group: String,
    pub prompt: String,
    pub context_tokens: usize,
    pub output_tokens: u32,
    pub concurrency: u32,
    pub temperature: f32,
    pub seed: u64,
    pub minimum_speedup: f64,
}

impl Suite {
    pub(super) fn load(path: &Path, names: &[String], group: &str) -> Result<Self, Box<dyn Error>> {
        let mut suite: Self = serde_json::from_slice(&fs::read(path)?)?;
        if suite.schema_version != 1
            || suite.source.is_empty()
            || suite.target.is_empty()
            || suite.padding.is_empty()
            || suite.context_semantics.is_empty()
            || !suite.policy.is_object()
        {
            return Err("invalid completion workload provenance or schema".into());
        }
        let mut found = BTreeSet::new();
        for row in &suite.rows {
            if !found.insert(row.name.as_str())
                || row.output_tokens == 0
                || row.concurrency == 0
                || !row.temperature.is_finite()
                || row.temperature < 0.0
                || !row.minimum_speedup.is_finite()
                || row.minimum_speedup <= 0.0
                || !matches!(row.group.as_str(), "regular" | "long")
            {
                return Err("invalid or duplicate completion workload row".into());
            }
        }
        if names.iter().any(|name| !found.contains(name.as_str())) {
            return Err("unknown completion workload row".into());
        }
        suite
            .rows
            .retain(|row| row.group == group && (names.is_empty() || names.contains(&row.name)));
        if suite.rows.is_empty()
            || (!names.is_empty()
                && suite.rows.len() != names.iter().collect::<BTreeSet<_>>().len())
        {
            return Err("completion workload has no rows".into());
        }
        Ok(suite)
    }
}

pub(super) fn render(
    tokenizer: &Tokenizer,
    prompt: &str,
    padding: &str,
    context: usize,
    wrapper: (&str, &str),
) -> Result<String, Box<dyn Error>> {
    let count = |text: &str| {
        tokenizer
            .encode(text, false)
            .map(|tokens| tokens.len())
            .map_err(|error| format!("cannot tokenize workload: {error}"))
    };
    if context == 0 || count(prompt)? >= context {
        return Ok(prompt.to_owned());
    }
    if padding.is_empty() || !padding.is_ascii() {
        return Err("context padding must be nonempty ASCII".into());
    }
    let (header, footer) = wrapper;
    let wrap = |prefix: &str| format!("{header}{prefix}{footer}{prompt}");
    if count(&wrap(""))? > context {
        return Err("context budget cannot fit the task wrapper".into());
    }
    if count(&wrap(padding))? < context {
        return Err("distinct padding corpus cannot fill the context budget".into());
    }
    let (mut lo, mut hi) = (0, padding.len());
    while lo < hi {
        let middle = lo + (hi - lo).div_ceil(2);
        if count(&wrap(&padding[..middle]))? <= context {
            lo = middle;
        } else {
            hi = middle - 1;
        }
    }
    Ok(wrap(&padding[..lo]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_and_named_rows_are_selected_before_execution() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/suites/prompt-lookup.json");
        let regular = Suite::load(&path, &[], "regular").unwrap();
        assert!(regular.rows.iter().all(|row| row.group == "regular"));
        assert_eq!(regular.padding_prefix, "<|im_start|>system\n");
        assert_eq!(regular.padding_suffix, "<|im_end|>\n");
        let long = Suite::load(&path, &["chat-32k".to_owned()], "long").unwrap();
        assert_eq!(long.rows.len(), 1);
        assert!(Suite::load(&path, &["chat-32k".to_owned()], "regular").is_err());
        assert!(Suite::load(&path, &["unknown".to_owned()], "regular").is_err());
    }

    #[test]
    fn context_padding_preserves_the_task_and_respects_the_token_budget() {
        let vocab = [
            ("[UNK]".to_owned(), 0),
            ("history".to_owned(), 1),
            ("task".to_owned(), 2),
        ];
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into())
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let prompt = "task task";
        assert_eq!(
            render(&tokenizer, prompt, "history ", 0, ("header ", "footer ")).unwrap(),
            prompt
        );
        let rendered = render(
            &tokenizer,
            prompt,
            &"history ".repeat(64),
            33,
            ("header ", "footer "),
        )
        .unwrap();
        assert!(rendered.ends_with(prompt));
        assert!(tokenizer.encode(rendered, false).unwrap().len() <= 33);
        assert!(render(&tokenizer, prompt, "history ", 33, ("header ", "footer ")).is_err());
        assert!(render(&tokenizer, prompt, "", 33, ("header ", "footer ")).is_err());
        assert!(render(&tokenizer, prompt, "history ", 3, ("header ", "footer ")).is_err());
    }
}
