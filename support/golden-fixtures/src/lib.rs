#![warn(missing_docs)]
#![forbid(unsafe_code)]

//! Verified readers and numeric comparisons for transformer golden fixtures.

mod comparison;

pub use comparison::{
    BF16_HIDDEN_STATE_TOLERANCE, BF16_LOGIT_KL_TOLERANCE, ComparisonError, LOGIT_KL_TOLERANCE,
    LayerComparison, compare_hidden_states, mean_logit_kl_divergence, normwise_relative_error,
};

use std::{error::Error, fmt, fs, path::Path};

use safetensors::{SafeTensors, tensor::Dtype};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// A validated owned float32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl FloatTensor {
    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the contiguous row-major values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// All validated tensors for one prompt.
#[derive(Clone, Debug)]
pub struct PromptFixture {
    name: String,
    text: String,
    prompt_ids: Vec<i64>,
    hidden_states: Vec<FloatTensor>,
    router_logits: Option<FloatTensor>,
    prompt_logits: FloatTensor,
    greedy_tokens: Vec<i64>,
    greedy_step_logits: FloatTensor,
}

impl PromptFixture {
    /// Returns the stable prompt name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the raw prompt text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the raw prompt token ids.
    #[must_use]
    pub fn prompt_ids(&self) -> &[i64] {
        &self.prompt_ids
    }

    /// Returns embeddings followed by every per-layer hidden state.
    #[must_use]
    pub fn hidden_states(&self) -> &[FloatTensor] {
        &self.hidden_states
    }

    /// Returns one indexed hidden state, where zero is the embeddings.
    #[must_use]
    pub fn hidden_state(&self, index: usize) -> Option<&FloatTensor> {
        self.hidden_states.get(index)
    }

    /// Returns prompt-pass router logits as layers by tokens by experts.
    #[must_use]
    pub const fn router_logits(&self) -> Option<&FloatTensor> {
        self.router_logits.as_ref()
    }

    /// Returns the last-position prompt logits.
    #[must_use]
    pub fn prompt_logits(&self) -> &FloatTensor {
        &self.prompt_logits
    }

    /// Returns the 32 greedy-decoded token ids.
    #[must_use]
    pub fn greedy_tokens(&self) -> &[i64] {
        &self.greedy_tokens
    }

    /// Returns the logits used to select every greedy token.
    #[must_use]
    pub fn greedy_step_logits(&self) -> &FloatTensor {
        &self.greedy_step_logits
    }
}

/// A verified collection of transformer fixtures.
#[derive(Clone, Debug)]
pub struct FixtureDirectory {
    model_sha256: String,
    prompts: Vec<PromptFixture>,
}

impl FixtureDirectory {
    /// Loads the manifest and verifies every fixture hash and tensor shape.
    ///
    /// # Errors
    ///
    /// Returns an error for unreadable, malformed, corrupted, or inconsistent fixtures.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FixtureError> {
        let path = path.as_ref();
        let manifest = fs::read(path.join("manifest.json")).map_err(FixtureError::Io)?;
        let manifest = serde_json::from_slice::<Manifest>(&manifest)
            .map_err(|error| FixtureError::Manifest(error.to_string()))?;
        if manifest.schema_version != 1 {
            return Err(FixtureError::UnsupportedSchema(manifest.schema_version));
        }
        let mut prompts = Vec::with_capacity(manifest.prompts.len());
        for prompt in manifest.prompts {
            if prompts
                .iter()
                .any(|fixture: &PromptFixture| fixture.name == prompt.name)
            {
                return Err(FixtureError::DuplicatePrompt(prompt.name));
            }
            prompts.push(load_prompt(path, prompt)?);
        }
        Ok(Self {
            model_sha256: manifest.model.sha256,
            prompts,
        })
    }

    /// Returns the SHA-256 digest of the source model file.
    #[must_use]
    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    /// Returns every prompt fixture in manifest order.
    #[must_use]
    pub fn prompts(&self) -> &[PromptFixture] {
        &self.prompts
    }

    /// Finds a prompt fixture by its stable name.
    #[must_use]
    pub fn prompt(&self, name: &str) -> Option<&PromptFixture> {
        self.prompts.iter().find(|prompt| prompt.name == name)
    }
}

/// Decodes contiguous little-endian f32 values.
///
/// # Errors
///
/// Returns an error when `bytes` ends with a partial f32 value.
pub fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>, DecodeError> {
    let (values, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(DecodeError);
    }
    Ok(values
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

/// Computes the lowercase SHA-256 digest of a file.
///
/// # Errors
///
/// Returns an error when the file cannot be read.
pub fn sha256_file(path: impl AsRef<Path>) -> std::io::Result<String> {
    fs::read(path).map(|bytes| format!("{:x}", Sha256::digest(bytes)))
}

/// A byte slice ended with a partial f32 value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError;

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("tensor contains a partial f32 value")
    }
}

impl Error for DecodeError {}

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u64,
    model: ModelMetadata,
    prompts: Vec<PromptMetadata>,
}

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct PromptMetadata {
    name: String,
    text: String,
    file: String,
    sha256: String,
    prompt_tokens: usize,
    router_logits: Option<RouterMetadata>,
}

#[derive(Debug, Deserialize)]
struct RouterMetadata {
    shape: Vec<usize>,
}

fn load_prompt(root: &Path, metadata: PromptMetadata) -> Result<PromptFixture, FixtureError> {
    let file_name = Path::new(&metadata.file);
    if file_name.file_name() != Some(file_name.as_os_str()) {
        return Err(FixtureError::InvalidFileName(metadata.file));
    }
    let bytes = fs::read(root.join(file_name)).map_err(FixtureError::Io)?;
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    if actual_hash != metadata.sha256 {
        return Err(FixtureError::HashMismatch {
            file: metadata.file,
            expected: metadata.sha256,
            actual: actual_hash,
        });
    }
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| FixtureError::Tensor(error.to_string()))?;
    let prompt_ids = read_i64(&tensors, "prompt_token_ids")?;
    if prompt_ids.len() != metadata.prompt_tokens {
        return Err(shape_error("prompt_token_ids"));
    }
    let hidden_state_count = tensors
        .names()
        .iter()
        .filter(|name| {
            name.strip_prefix("hidden_state_")
                .is_some_and(|index| index.parse::<usize>().is_ok())
        })
        .count();
    if hidden_state_count < 2 {
        return Err(shape_error("hidden_states"));
    }
    let mut hidden_states = Vec::with_capacity(hidden_state_count);
    for index in 0..hidden_state_count {
        let name = format!("hidden_state_{index}");
        let tensor = read_f32(&tensors, &name)?;
        if tensor.shape.len() != 2 || tensor.shape[0] != prompt_ids.len() {
            return Err(shape_error(&name));
        }
        if hidden_states
            .first()
            .is_some_and(|first: &FloatTensor| first.shape != tensor.shape)
        {
            return Err(shape_error(&name));
        }
        hidden_states.push(tensor);
    }
    let router_logits = match metadata.router_logits {
        Some(router) => {
            let tensor = read_f32(&tensors, "router_logits")?;
            if router.shape != tensor.shape
                || tensor.shape.len() != 3
                || tensor.shape[0] != hidden_states.len() - 1
                || tensor.shape[1] != prompt_ids.len()
                || tensor.shape[2] == 0
            {
                return Err(shape_error("router_logits"));
            }
            Some(tensor)
        }
        None if tensors.names().contains(&"router_logits") => {
            return Err(shape_error("router_logits"));
        }
        None => None,
    };
    let prompt_logits = read_f32(&tensors, "prompt_last_logits")?;
    let greedy_tokens = read_i64(&tensors, "greedy_token_ids")?;
    let greedy_step_logits = read_f32(&tensors, "greedy_step_logits")?;
    if prompt_logits.shape.len() != 1
        || greedy_step_logits.shape.as_slice() != [greedy_tokens.len(), prompt_logits.shape[0]]
    {
        return Err(shape_error("greedy_step_logits"));
    }
    Ok(PromptFixture {
        name: metadata.name,
        text: metadata.text,
        prompt_ids,
        hidden_states,
        router_logits,
        prompt_logits,
        greedy_tokens,
        greedy_step_logits,
    })
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<FloatTensor, FixtureError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|error| FixtureError::Tensor(error.to_string()))?;
    if tensor.dtype() != Dtype::F32 {
        return Err(FixtureError::WrongDtype(name.to_owned()));
    }
    let shape = tensor.shape().to_vec();
    validate_byte_len(name, &shape, tensor.data().len(), 4)?;
    let values =
        decode_f32_le(tensor.data()).map_err(|error| FixtureError::Tensor(error.to_string()))?;
    Ok(FloatTensor { shape, values })
}

fn read_i64(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i64>, FixtureError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|error| FixtureError::Tensor(error.to_string()))?;
    if tensor.dtype() != Dtype::I64 {
        return Err(FixtureError::WrongDtype(name.to_owned()));
    }
    if tensor.shape().len() != 1 {
        return Err(shape_error(name));
    }
    validate_byte_len(name, tensor.shape(), tensor.data().len(), 8)?;
    Ok(tensor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| {
            i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        })
        .collect())
}

fn validate_byte_len(
    name: &str,
    shape: &[usize],
    actual: usize,
    width: u64,
) -> Result<(), FixtureError> {
    let elements = shape.iter().try_fold(1_u64, |count, &extent| {
        u64::try_from(extent)
            .ok()
            .and_then(|extent| count.checked_mul(extent))
    });
    let expected = elements.and_then(|count| count.checked_mul(width));
    if expected != u64::try_from(actual).ok() {
        return Err(shape_error(name));
    }
    Ok(())
}

fn shape_error(name: &str) -> FixtureError {
    FixtureError::InvalidShape(name.to_owned())
}

/// A fixture loading failure.
#[derive(Debug)]
pub enum FixtureError {
    /// A file could not be read.
    Io(std::io::Error),
    /// The manifest is not valid JSON with the expected fields.
    Manifest(String),
    /// The manifest schema is not supported.
    UnsupportedSchema(u64),
    /// Two prompts have the same stable name.
    DuplicatePrompt(String),
    /// A fixture path is not a plain file name.
    InvalidFileName(String),
    /// A fixture does not match its manifest digest.
    HashMismatch {
        /// The fixture file name.
        file: String,
        /// The digest recorded in the manifest.
        expected: String,
        /// The digest computed from the file.
        actual: String,
    },
    /// A safetensors file or tensor name is invalid.
    Tensor(String),
    /// A tensor has an unexpected scalar type.
    WrongDtype(String),
    /// A tensor has an invalid or inconsistent shape.
    InvalidShape(String),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "fixture I/O failed: {error}"),
            Self::Manifest(error) => write!(formatter, "invalid fixture manifest: {error}"),
            Self::UnsupportedSchema(version) => {
                write!(formatter, "unsupported fixture schema {version}")
            }
            Self::DuplicatePrompt(name) => write!(formatter, "duplicate prompt {name:?}"),
            Self::InvalidFileName(name) => write!(formatter, "invalid fixture file name {name:?}"),
            Self::HashMismatch {
                file,
                expected,
                actual,
            } => write!(
                formatter,
                "fixture {file:?} has SHA-256 {actual}, expected {expected}"
            ),
            Self::Tensor(error) => write!(formatter, "invalid fixture tensors: {error}"),
            Self::WrongDtype(name) => write!(formatter, "tensor {name:?} has the wrong dtype"),
            Self::InvalidShape(name) => write!(formatter, "tensor {name:?} has an invalid shape"),
        }
    }
}

impl Error for FixtureError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::{Path, PathBuf},
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use safetensors::tensor::{TensorView, serialize};
    use serde_json::{Value, json};

    use super::*;

    #[derive(Clone, Copy)]
    enum TensorMutation {
        None,
        WrongDtype,
        InvalidShape,
        InvalidRouterShape,
    }

    struct TensorBytes {
        name: String,
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                env::temp_dir().join(format!("forja-golden-fixtures-{}-{nonce}", process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn case(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir(&path).unwrap();
            path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn reader_refuses_invalid_fixture_boundaries() {
        let temporary = TestDirectory::new();

        let case = temporary.case("valid-router");
        let hash = write_tensors(&case, TensorMutation::None);
        write_manifest(&case, 1, &[prompt("fixture.safetensors", &hash)]);
        let fixtures = FixtureDirectory::open(case).unwrap();
        assert_eq!(
            fixtures.prompts()[0].router_logits().unwrap().shape(),
            [2, 1, 3]
        );

        let case = temporary.case("hash-mismatch");
        let hash = write_tensors(&case, TensorMutation::None);
        write_manifest(&case, 1, &[prompt("fixture.safetensors", &hash)]);
        let manifest = case.join("manifest.json");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(&manifest, contents.replace(&hash, &"0".repeat(64))).unwrap();
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::HashMismatch { .. }
        ));

        let case = temporary.case("wrong-dtype");
        let hash = write_tensors(&case, TensorMutation::WrongDtype);
        write_manifest(&case, 1, &[prompt("fixture.safetensors", &hash)]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::WrongDtype(name) if name == "prompt_token_ids"
        ));

        let case = temporary.case("invalid-shape");
        let hash = write_tensors(&case, TensorMutation::InvalidShape);
        write_manifest(&case, 1, &[prompt("fixture.safetensors", &hash)]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::InvalidShape(name) if name == "hidden_state_0"
        ));

        let case = temporary.case("invalid-router-shape");
        let hash = write_tensors(&case, TensorMutation::InvalidRouterShape);
        write_manifest(&case, 1, &[prompt("fixture.safetensors", &hash)]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::InvalidShape(name) if name == "router_logits"
        ));

        let case = temporary.case("duplicate-prompt");
        let hash = write_tensors(&case, TensorMutation::None);
        let duplicate = prompt("fixture.safetensors", &hash);
        write_manifest(&case, 1, &[duplicate.clone(), duplicate]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::DuplicatePrompt(name) if name == "tiny"
        ));

        let case = temporary.case("invalid-filename");
        write_manifest(&case, 1, &[prompt("../fixture.safetensors", "unused")]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::InvalidFileName(name) if name == "../fixture.safetensors"
        ));

        let case = temporary.case("unsupported-schema");
        write_manifest(&case, 2, &[]);
        assert!(matches!(
            FixtureDirectory::open(case).unwrap_err(),
            FixtureError::UnsupportedSchema(2)
        ));
    }

    fn prompt(file: &str, sha256: &str) -> Value {
        json!({
            "name": "tiny",
            "text": "x",
            "file": file,
            "sha256": sha256,
            "prompt_tokens": 1,
            "router_logits": { "shape": [2, 1, 3] }
        })
    }

    fn write_manifest(path: &Path, schema_version: u64, prompts: &[Value]) {
        let manifest = json!({
            "schema_version": schema_version,
            "model": { "sha256": "model" },
            "prompts": prompts
        });
        fs::write(
            path.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn write_tensors(path: &Path, mutation: TensorMutation) -> String {
        let mut tensors = vec![TensorBytes {
            name: "prompt_token_ids".to_owned(),
            dtype: if matches!(mutation, TensorMutation::WrongDtype) {
                Dtype::F32
            } else {
                Dtype::I64
            },
            shape: vec![1],
            data: if matches!(mutation, TensorMutation::WrongDtype) {
                vec![0; 4]
            } else {
                vec![0; 8]
            },
        }];
        for index in 0..3 {
            tensors.push(TensorBytes {
                name: format!("hidden_state_{index}"),
                dtype: Dtype::F32,
                shape: if index == 0 && matches!(mutation, TensorMutation::InvalidShape) {
                    vec![1]
                } else {
                    vec![1, 1]
                },
                data: vec![0; 4],
            });
        }
        tensors.extend([
            TensorBytes {
                name: "router_logits".to_owned(),
                dtype: Dtype::F32,
                shape: if matches!(mutation, TensorMutation::InvalidRouterShape) {
                    vec![2, 2, 3]
                } else {
                    vec![2, 1, 3]
                },
                data: if matches!(mutation, TensorMutation::InvalidRouterShape) {
                    vec![0; 48]
                } else {
                    vec![0; 24]
                },
            },
            TensorBytes {
                name: "prompt_last_logits".to_owned(),
                dtype: Dtype::F32,
                shape: vec![2],
                data: vec![0; 8],
            },
            TensorBytes {
                name: "greedy_token_ids".to_owned(),
                dtype: Dtype::I64,
                shape: vec![1],
                data: vec![0; 8],
            },
            TensorBytes {
                name: "greedy_step_logits".to_owned(),
                dtype: Dtype::F32,
                shape: vec![1, 2],
                data: vec![0; 8],
            },
        ]);
        let views = tensors.iter().map(|tensor| {
            (
                tensor.name.clone(),
                TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
            )
        });
        let bytes = serialize(views, None).unwrap();
        fs::write(path.join("fixture.safetensors"), &bytes).unwrap();
        format!("{:x}", Sha256::digest(bytes))
    }
}
