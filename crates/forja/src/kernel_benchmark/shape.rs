use forja_core::{Backend, DType, Op, Tensor, ViewOp};
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Shape {
    #[serde(with = "Operation")]
    pub(super) operation: Op,
    pub(super) inputs: Vec<TensorShape>,
    pub(super) outputs: Vec<TensorShape>,
}

#[derive(Deserialize)]
#[serde(remote = "Op", tag = "kind", rename_all = "kebab-case")]
enum Operation {
    Matmul,
    QuantMatmul {
        bits: u8,
        group_size: u32,
    },
    GatherQuantMatmul {
        bits: u8,
        group_size: u32,
    },
    Sdpa {
        scale: f32,
        causal: bool,
        q_start: u32,
    },
    PagedSdpaAppend {
        scale: f32,
        q_start: u32,
        page_tokens: u32,
    },
    PagedSdpa {
        scale: f32,
        causal: bool,
        q_start: u32,
        key_length: u32,
        page_tokens: u32,
    },
    TopK {
        k: u32,
        normalize: bool,
        chunk_width: u32,
    },
    Sample {
        position: u32,
        chunk_width: u32,
        fast_top_k: u32,
        fast_max_chunks: u32,
        rejection_rounds: u32,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TensorShape {
    dtype: String,
    shape: Vec<u32>,
    #[serde(default)]
    u32_values: Option<Vec<u32>>,
    #[serde(default)]
    permutation: Option<Vec<u8>>,
}

impl TensorShape {
    pub(super) fn layout(&self) -> Result<(DType, u64), Box<dyn Error>> {
        let (dtype, size) = match self.dtype.as_str() {
            "f32" => (DType::F32, 4_u64),
            "f16" => (DType::F16, 2),
            "bf16" => (DType::BF16, 2),
            "u32" => (DType::U32, 4),
            _ => return Err("unsupported tensor dtype".into()),
        };
        let bytes = self.shape.iter().try_fold(size, |bytes, &dimension| {
            bytes
                .checked_mul(u64::from(dimension))
                .ok_or("tensor size overflow")
        })?;
        if let Some(values) = &self.u32_values
            && (dtype != DType::U32 || u64::try_from(values.len())?.checked_mul(4) != Some(bytes))
        {
            return Err("u32_values must exactly fill a u32 tensor".into());
        }
        Ok((dtype, bytes))
    }

    pub(super) fn allocate(&self, backend: &impl Backend) -> Result<Tensor, Box<dyn Error>> {
        let (dtype, bytes) = self.layout()?;
        let tensor = backend.alloc(dtype, &self.shape)?;
        let data = self.u32_values.as_ref().map_or_else(
            || Ok(vec![0; usize::try_from(bytes)?]),
            |values| Ok::<_, Box<dyn Error>>(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        )?;
        backend.write(&tensor, &data)?;
        if let Some(axes) = &self.permutation {
            return Ok(backend.view(&tensor, ViewOp::Permute(axes.clone()))?);
        }
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shapes_refuse_overflow_and_mismatched_metadata() {
        let spec: TensorShape =
            serde_json::from_str(r#"{"dtype":"f32","shape":[4294967295,4294967295,7]}"#).unwrap();
        assert!(spec.layout().is_err());
        let spec: TensorShape =
            serde_json::from_str(r#"{"dtype":"u32","shape":[7],"u32_values":[1]}"#).unwrap();
        assert!(spec.layout().is_err());
    }

    #[test]
    fn paged_shapes_preserve_noncontiguous_physical_order() {
        let backend = forja_cpu::CpuBackend::new();
        let spec: TensorShape =
            serde_json::from_str(r#"{"dtype":"u32","shape":[3,2],"u32_values":[2,1,0,1,1,1]}"#)
                .unwrap();
        let tensor = spec.allocate(&backend).unwrap();
        assert_eq!(
            backend.read(&tensor).unwrap(),
            [2_u32, 1, 0, 1, 1, 1]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn operation_and_tensor_specs_build_a_valid_core_dispatch() {
        let shape: Shape = serde_json::from_str(r#"{"operation":{"kind":"matmul"},"inputs":[{"dtype":"f32","shape":[1,7]},{"dtype":"f32","shape":[7,33]}],"outputs":[{"dtype":"f32","shape":[1,33]}]}"#).unwrap();
        let backend = forja_cpu::CpuBackend::new();
        let inputs = shape
            .inputs
            .iter()
            .map(|s| s.allocate(&backend).unwrap())
            .collect::<Vec<_>>();
        let outputs = shape
            .outputs
            .iter()
            .map(|s| s.allocate(&backend).unwrap())
            .collect::<Vec<_>>();
        forja_core::CommandList::new()
            .dispatch_many(
                shape.operation,
                &inputs.iter().collect::<Vec<_>>(),
                &outputs.iter().collect::<Vec<_>>(),
            )
            .unwrap();
    }
}
