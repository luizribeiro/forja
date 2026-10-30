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
    #[serde(default)]
    pub(super) epilogue: Option<DiagnosticEpilogue>,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum DiagnosticEpilogue {
    Identity,
    SiluMul,
}

impl Shape {
    pub(super) fn prepare_epilogue(
        &self,
    ) -> Result<Option<std::sync::Arc<forja_core::program::PreparedEpilogue>>, Box<dyn Error>> {
        use forja_core::program::{
            BinOp, EpilogueSignature, Inst, Program, ProgramKind, UnOp, prepare_epilogue,
        };
        let Some(kind) = &self.epilogue else {
            return Ok(None);
        };
        let output = self.outputs.first().ok_or("epilogue requires an output")?;
        if self.outputs.len() != 1 {
            return Err("epilogue requires one output".into());
        }
        let (accumulators, insts) = match kind {
            DiagnosticEpilogue::Identity => (1, vec![Inst::Input(0)]),
            DiagnosticEpilogue::SiluMul => (
                2,
                vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Unary(UnOp::Neg, 0),
                    Inst::Unary(UnOp::Exp, 2),
                    Inst::Const(1.0),
                    Inst::Binary(BinOp::Add, 4, 3),
                    Inst::Binary(BinOp::Div, 0, 5),
                    Inst::Binary(BinOp::Mul, 6, 1),
                ],
            ),
        };
        let last = u32::try_from(insts.len() - 1)?;
        let program = Program {
            kind: ProgramKind::Map,
            insts,
            outputs: vec![(0, last)],
        }
        .validate()?;
        let signature = EpilogueSignature::new(
            accumulators,
            u8::try_from(output.shape.len())?,
            vec![],
            output.layout()?.0,
        );
        Ok(Some(prepare_epilogue(program, signature)?))
    }
}

#[derive(Deserialize)]
#[serde(remote = "Op", tag = "kind", rename_all = "kebab-case")]
enum Operation {
    State(#[serde(deserialize_with = "state_operation")] forja_core::StateOp),
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

fn state_operation<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<forja_core::StateOp, D::Error> {
    #[derive(Deserialize)]
    #[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
    enum StateOperation {
        DeltaRuleBlock,
    }
    use forja_core::{StateAccess, StateLayout, StatePlane, StatePool};
    let StateOperation::DeltaRuleBlock = StateOperation::deserialize(deserializer)?;
    let access = || -> Result<_, Box<dyn Error>> {
        let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![1])?])?;
        let pool = StatePool::new(layout, 1, 128)?;
        let mut sequence = pool.reserve()?;
        let transaction = sequence.begin(1)?;
        let slot = sequence.bind(&transaction)?;
        Ok(StateAccess::from_slot(&slot)?)
    };
    access()
        .map(forja_core::StateOp::DeltaRuleBlock)
        .map_err(serde::de::Error::custom)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TensorShape {
    dtype: String,
    shape: Vec<u32>,
    #[serde(default)]
    u32_values: Option<Vec<u32>>,
    #[serde(default)]
    seed: Option<u32>,
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
        if self.seed.is_some() && self.u32_values.is_some() {
            return Err("seed and explicit u32_values are mutually exclusive".into());
        }
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
        let byte_count = usize::try_from(bytes)?;
        let data = self.u32_values.as_ref().map_or_else(
            || {
                Ok(self.seed.map_or_else(
                    || vec![0; byte_count],
                    |seed| seeded_bytes(dtype, byte_count, seed),
                ))
            },
            |values| Ok::<_, Box<dyn Error>>(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        )?;
        backend.write(&tensor, &data)?;
        if let Some(axes) = &self.permutation {
            return Ok(backend.view(&tensor, ViewOp::Permute(axes.clone()))?);
        }
        Ok(tensor)
    }
}

fn seeded_bytes(dtype: DType, bytes: usize, mut seed: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity(bytes);
    while data.len() < bytes {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let [lo, hi, _, _] = seed.to_le_bytes();
        let low = u16::from_le_bytes([lo, hi]);
        match dtype {
            DType::U32 | DType::I32 => data.extend_from_slice(&seed.to_le_bytes()),
            DType::F32 => data.extend_from_slice(
                &(f32::from_bits(0x3f00_0000 | (seed & 0x007f_ffff)) - 0.75).to_le_bytes(),
            ),
            DType::F16 => data.extend_from_slice(&(0x3800_u16 | (low & 0x03ff)).to_le_bytes()),
            DType::BF16 => data.extend_from_slice(&(0x3f00_u16 | (low & 0x007f)).to_le_bytes()),
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seeded_operands_are_repeatable_and_nonzero() {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::U32] {
            let data = seeded_bytes(dtype, 132, 7);
            assert_eq!(data.len(), 132);
            assert_eq!(data, seeded_bytes(dtype, 132, 7));
            assert_ne!(data, seeded_bytes(dtype, 132, 8));
            assert!(data.iter().any(|&byte| byte != 0));
        }
        let spec: TensorShape =
            serde_json::from_str(r#"{"dtype":"u32","shape":[1],"u32_values":[1],"seed":7}"#)
                .unwrap();
        assert!(spec.layout().is_err());
    }

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
