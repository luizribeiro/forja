use forja_sdk::bf16;
use std::{error::Error, fs, path::PathBuf};
#[derive(Default)]
pub(super) struct Weights {
    header: serde_json::Map<String, serde_json::Value>,
    data: Vec<u8>,
}
impl Weights {
    pub(super) fn tensor(&mut self, name: &str, dtype: &str, shape: &[u32], data: &[u8]) {
        let start = self.data.len();
        self.data.extend(data);
        self.header.insert(
            name.to_owned(),
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,self.data.len()]}),
        );
    }
    pub(super) fn bf16(&mut self, name: &str, shape: &[u32], values: &[f32]) {
        self.tensor(
            name,
            "BF16",
            shape,
            &values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        );
    }
    pub(super) fn save(&self, name: &str) -> Result<PathBuf, Box<dyn Error>> {
        let mut header = serde_json::to_vec(&self.header)?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(&self.data);
        let path = std::env::temp_dir().join(format!("forja-qwen38-{name}-{}", std::process::id()));
        fs::write(&path, bytes)?;
        Ok(path)
    }
}
