use std::{
    collections::HashSet,
    error::Error,
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use forja_core::{DType, Layout, LayoutError, MappedRegion};
use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};
use serde_json::Value;

const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

/// Metadata for one tensor in a weight source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightTensor {
    name: String,
    dtype: DType,
    shape: Vec<u32>,
    byte_offset: u64,
    byte_len: u64,
}

impl WeightTensor {
    /// Returns the tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the tensor scalar type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Returns the absolute byte offset in the source file.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Returns the tensor byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Reports whether the tensor begins at its scalar type's alignment.
    #[must_use]
    pub const fn is_aligned(&self) -> bool {
        self.byte_offset.is_multiple_of(self.dtype.byte_size())
    }

    pub(crate) fn layout(&self, buffer_len: u64) -> Result<Layout, LayoutError> {
        Layout::contiguous(
            self.dtype,
            self.byte_offset / self.dtype.byte_size(),
            self.shape.clone(),
            buffer_len,
        )
    }
}

/// A validated collection of named model tensors.
pub trait WeightSource {
    /// Returns every tensor's name, scalar type, and shape.
    fn tensors(&self) -> &[WeightTensor];
    /// Maps the source bytes for import into a backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the source file can no longer be mapped.
    fn mapped_region(&self) -> Result<MappedRegion, WeightError>;
}

/// A validated safetensors file.
#[derive(Debug)]
pub struct Safetensors {
    path: PathBuf,
    tensors: Vec<WeightTensor>,
}

impl Safetensors {
    /// Opens and validates a safetensors file containing supported scalar types.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed metadata, unsupported scalar types, or invalid byte ranges.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(WeightError::Io)?;
        let file_len = file.metadata().map_err(WeightError::Io)?.len();
        let header_len = read_header_len(&mut file)?;
        if header_len > MAX_HEADER_BYTES {
            return Err(WeightError::HeaderTooLarge(header_len));
        }
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| WeightError::HeaderTooLarge(header_len))?;
        let mut header = Vec::new();
        header
            .try_reserve_exact(header_len_usize)
            .map_err(|_| WeightError::HeaderTooLarge(header_len))?;
        header.resize(header_len_usize, 0);
        file.read_exact(&mut header)
            .map_err(|_| WeightError::Truncated)?;
        let data_start = 8_u64
            .checked_add(header_len)
            .ok_or(WeightError::InvalidRange)?;
        let entries = serde_json::from_slice::<HeaderEntries>(&header)
            .map_err(|error| WeightError::InvalidHeader(error.to_string()))?;
        let mut tensors = entries
            .0
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, value)| parse_tensor(name, &value, data_start, file_len))
            .collect::<Result<Vec<_>, _>>()?;
        tensors.sort_by_key(WeightTensor::byte_offset);
        if tensors.windows(2).any(|pair| {
            pair[0]
                .byte_offset
                .checked_add(pair[0].byte_len)
                .is_none_or(|end| end > pair[1].byte_offset)
        }) {
            return Err(WeightError::OverlappingRanges);
        }
        Ok(Self {
            path: path.to_owned(),
            tensors,
        })
    }
}

struct HeaderEntries(Vec<(String, Value)>);

impl<'de> Deserialize<'de> for HeaderEntries {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct HeaderVisitor;

impl<'de> Visitor<'de> for HeaderVisitor {
    type Value = HeaderEntries;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors metadata object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut names = HashSet::new();
        let mut entries = Vec::new();
        while let Some(name) = map.next_key::<String>()? {
            if !names.insert(name.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate safetensors entry {name:?}"
                )));
            }
            entries.push((name, map.next_value()?));
        }
        Ok(HeaderEntries(entries))
    }
}

impl WeightSource for Safetensors {
    fn tensors(&self) -> &[WeightTensor] {
        &self.tensors
    }

    fn mapped_region(&self) -> Result<MappedRegion, WeightError> {
        let file = File::open(&self.path).map_err(WeightError::Io)?;
        MappedRegion::map(&file).map_err(WeightError::Mapping)
    }
}

/// A reason a weight source could not be opened.
#[derive(Debug)]
pub enum WeightError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The validated file could not be mapped.
    Mapping(std::io::Error),
    /// The file ended before its declared metadata or tensor data.
    Truncated,
    /// The JSON metadata is malformed or has an invalid schema.
    InvalidHeader(String),
    /// The declared JSON metadata exceeds the parser's bound.
    HeaderTooLarge(u64),
    /// A tensor uses a scalar type unsupported by plain weights.
    UnsupportedDtype(String),
    /// A tensor byte range is invalid or inconsistent with its shape.
    InvalidRange,
    /// Two tensor byte ranges overlap.
    OverlappingRanges,
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "weight file I/O failed: {error}"),
            Self::Mapping(error) => error.fmt(f),
            Self::Truncated => f.write_str("weight file is truncated"),
            Self::InvalidHeader(error) => write!(f, "invalid safetensors header: {error}"),
            Self::HeaderTooLarge(bytes) => {
                write!(
                    f,
                    "safetensors header has {bytes} bytes, exceeding the size cap"
                )
            }
            Self::UnsupportedDtype(dtype) => {
                write!(f, "unsupported safetensors dtype {dtype}")
            }
            Self::InvalidRange => f.write_str("invalid safetensors tensor byte range"),
            Self::OverlappingRanges => f.write_str("safetensors tensor byte ranges overlap"),
        }
    }
}

impl Error for WeightError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) | Self::Mapping(error) => Some(error),
            _ => None,
        }
    }
}

fn read_header_len(file: &mut File) -> Result<u64, WeightError> {
    let mut bytes = [0; 8];
    file.read_exact(&mut bytes)
        .map_err(|_| WeightError::Truncated)?;
    Ok(u64::from_le_bytes(bytes))
}

fn parse_tensor(
    name: String,
    value: &Value,
    data_start: u64,
    file_len: u64,
) -> Result<WeightTensor, WeightError> {
    let entry = value
        .as_object()
        .ok_or_else(|| invalid_header(&name, "entry is not an object"))?;
    let dtype_name = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_header(&name, "dtype is missing"))?;
    let dtype = match dtype_name {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "U32" => DType::U32,
        other => return Err(WeightError::UnsupportedDtype(other.to_owned())),
    };
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_header(&name, "shape is missing"))?
        .iter()
        .map(|extent| {
            extent
                .as_u64()
                .and_then(|extent| u32::try_from(extent).ok())
                .ok_or_else(|| invalid_header(&name, "shape extent is not a u32"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let range = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|range| range.len() == 2)
        .ok_or_else(|| invalid_header(&name, "data_offsets must contain two values"))?;
    let relative_start = range[0]
        .as_u64()
        .ok_or_else(|| invalid_header(&name, "data offset is not a u64"))?;
    let relative_end = range[1]
        .as_u64()
        .ok_or_else(|| invalid_header(&name, "data offset is not a u64"))?;
    let byte_len = relative_end
        .checked_sub(relative_start)
        .ok_or(WeightError::InvalidRange)?;
    let elements = shape
        .iter()
        .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
    let expected = elements
        .and_then(|count| count.checked_mul(dtype.byte_size()))
        .ok_or(WeightError::InvalidRange)?;
    if byte_len != expected {
        return Err(WeightError::InvalidRange);
    }
    let byte_offset = data_start
        .checked_add(relative_start)
        .ok_or(WeightError::InvalidRange)?;
    let byte_end = byte_offset
        .checked_add(byte_len)
        .ok_or(WeightError::InvalidRange)?;
    if byte_end > file_len {
        return Err(WeightError::Truncated);
    }
    Ok(WeightTensor {
        name,
        dtype,
        shape,
        byte_offset,
        byte_len,
    })
}

fn invalid_header(name: &str, message: &str) -> WeightError {
    WeightError::InvalidHeader(format!("tensor {name:?}: {message}"))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn file(header: &str, data: &[u8], pad: bool) -> PathBuf {
        let mut header = header.as_bytes().to_vec();
        if pad {
            while !(header.len() + 8).is_multiple_of(8) {
                header.push(b' ');
            }
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        let path = std::env::temp_dir().join(format!(
            "forja-safetensors-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn open(header: &str, data: &[u8], pad: bool) -> Result<Safetensors, WeightError> {
        let path = file(header, data, pad);
        let result = Safetensors::open(&path);
        fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn parses_supported_tensor_metadata() {
        let source = open(
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"c":{"dtype":"U32","shape":[1],"data_offsets":[8,12]}}"#,
            &[0; 12],
            true,
        )
        .unwrap();

        assert_eq!(source.tensors().len(), 3);
        assert_eq!(source.tensors()[0].name(), "a");
        assert_eq!(source.tensors()[0].dtype(), DType::F32);
        assert_eq!(source.tensors()[1].shape(), [2]);
        assert_eq!(source.tensors()[2].dtype(), DType::U32);
    }

    #[test]
    fn rejects_truncated_tensor_data() {
        assert!(matches!(
            open(
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                &[0; 4],
                true
            ),
            Err(WeightError::Truncated)
        ));
    }

    #[test]
    fn rejects_overlapping_ranges() {
        assert!(matches!(
            open(
                r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                &[0; 8],
                true
            ),
            Err(WeightError::OverlappingRanges)
        ));
    }

    #[test]
    fn retains_unaligned_tensor_metadata_for_copying() {
        let source = open(
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
            false,
        )
        .unwrap();

        assert!(!source.tensors()[0].is_aligned());
    }

    #[test]
    fn rejects_ranges_past_the_file_end() {
        assert!(matches!(
            open(
                r#"{"a":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#,
                &[0; 6],
                true
            ),
            Err(WeightError::Truncated)
        ));
    }

    #[test]
    fn rejects_unknown_dtypes() {
        assert!(matches!(
            open(
                r#"{"a":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}"#,
                &[0],
                true
            ),
            Err(WeightError::UnsupportedDtype(dtype)) if dtype == "F8_E4M3"
        ));
    }

    #[test]
    fn rejects_duplicate_tensor_names() {
        assert!(matches!(
            open(
                r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 4],
                true
            ),
            Err(WeightError::InvalidHeader(_))
        ));
    }

    #[test]
    fn rejects_deeply_nested_headers() {
        let header = format!("{{\"a\":{}0{}}}", "[".repeat(256), "]".repeat(256));
        assert!(matches!(
            open(&header, &[], true),
            Err(WeightError::InvalidHeader(_))
        ));
    }

    #[test]
    fn rejects_headers_over_the_size_cap_before_reading_them() {
        let path = std::env::temp_dir().join(format!(
            "forja-safetensors-cap-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&path, (MAX_HEADER_BYTES + 1).to_le_bytes()).unwrap();
        let result = Safetensors::open(&path);
        fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(WeightError::HeaderTooLarge(_))));
    }
}
