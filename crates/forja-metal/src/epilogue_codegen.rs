use std::fmt::{self, Write};

use forja_core::{
    DType,
    program::{PreparedEpilogue, ProgramHash, ValueType},
};

use crate::scalar_codegen;

pub(super) const EXTRA_CAPACITY: usize = 7;
const IDENTITY_HELPER: &str = r"inline float forja_epilogue(
    float accumulator0,
    float accumulator1,
    uint coordinate0,
    uint coordinate1,
    uint coordinate2,
    uint extent0,
    uint extent1,
    uint extent2,
    device const uchar *epilogue_input0,
    device const uchar *epilogue_input1,
    device const uchar *epilogue_input2,
    device const uchar *epilogue_input3,
    device const uchar *epilogue_input4,
    device const uchar *epilogue_input5,
    device const uchar *epilogue_input6,
    constant TensorLayout *epilogue_layouts) {
    return accumulator0;
}
";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub(super) enum FamilySourceTag {
    Dense,
    Quant,
    GatherQuant,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct SourceKey {
    pub(super) family: FamilySourceTag,
    pub(super) hash: ProgramHash,
    pub(super) accumulator_count: u8,
    pub(super) logical_rank: u8,
    pub(super) extra_dtypes: Vec<u8>,
    pub(super) output_dtype: u8,
}

impl SourceKey {
    pub(super) fn new(family: FamilySourceTag, epilogue: &PreparedEpilogue) -> Self {
        let signature = epilogue.signature();
        Self {
            family,
            hash: epilogue.validated().content_hash(),
            accumulator_count: signature.accumulator_count(),
            logical_rank: signature.logical_rank(),
            extra_dtypes: signature
                .extra_dtypes()
                .iter()
                .copied()
                .map(dtype_key)
                .collect(),
            output_dtype: dtype_key(signature.output_dtype()),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct PipelineKey {
    pub(super) source: SourceKey,
    pub(super) implementation: String,
    pub(super) constants: Vec<(u32, u32)>,
}

impl PipelineKey {
    pub(super) fn new(
        source: SourceKey,
        implementation: &str,
        mut constants: Vec<(u32, u32)>,
    ) -> Self {
        constants.sort_unstable();
        Self {
            source,
            implementation: implementation.to_owned(),
            constants,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum SourceError {
    IdentityHelperCount,
    AddressStructure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AddressFragment {
    slot: usize,
    expression: String,
}

impl AddressFragment {
    fn new(slot: usize, rank: usize) -> Self {
        let expression = (0..rank).fold(
            format!("epilogue_layouts[{slot}].offset"),
            |address, axis| {
                format!(
                    "{address} + ulong(coordinate{}) * epilogue_layouts[{slot}].strides[{axis}]",
                    coordinate_axis(rank, axis)
                )
            },
        );
        Self { slot, expression }
    }

    fn declaration(&self) -> String {
        format!("ulong epilogue_address{} = {};", self.slot, self.expression)
    }

    fn load(&self, dtype: DType) -> String {
        match dtype {
            DType::F32 | DType::F16 => format!(
                "load_float(epilogue_input{}, epilogue_address{}, {}u)",
                self.slot,
                self.slot,
                dtype_key(dtype)
            ),
            DType::BF16 => format!(
                "load_bfloat_bits(epilogue_input{}, epilogue_address{})",
                self.slot, self.slot
            ),
            DType::U32 => format!(
                "load_uint(epilogue_input{}, epilogue_address{})",
                self.slot, self.slot
            ),
            DType::I32 => unreachable!("validated epilogues reject i32 inputs"),
        }
    }
}

pub(super) fn generate_helper(epilogue: &PreparedEpilogue) -> Result<String, SourceError> {
    let signature = epilogue.signature();
    let accumulator_count = usize::from(signature.accumulator_count());
    let rank = usize::from(signature.logical_rank());
    let mut source = String::new();
    line(
        &mut source,
        format_args!(
            "inline float forja_epilogue(float accumulator0, float accumulator1, uint coordinate0, uint coordinate1, uint coordinate2, uint extent0, uint extent1, uint extent2,"
        ),
    );
    for slot in 0..EXTRA_CAPACITY {
        line(
            &mut source,
            format_args!("    device const uchar *epilogue_input{slot},"),
        );
    }
    line(
        &mut source,
        format_args!("    constant TensorLayout *epilogue_layouts) {{"),
    );
    let addresses = (0..signature.extra_dtypes().len())
        .map(|slot| AddressFragment::new(slot, rank))
        .collect::<Vec<_>>();
    for address in &addresses {
        line(&mut source, format_args!("    {}", address.declaration()));
    }

    let mut types = Vec::with_capacity(epilogue.validated().program().insts.len());
    for (index, &instruction) in epilogue.validated().program().insts.iter().enumerate() {
        let (value_type, expression) = scalar_codegen::instruction_expression(
            instruction,
            &types,
            |slot| {
                let slot = usize::try_from(slot).unwrap_or(usize::MAX);
                if slot < accumulator_count {
                    return (ValueType::F32, format!("accumulator{slot}"));
                }
                let extra = slot - accumulator_count;
                let dtype = signature.extra_dtypes()[extra];
                (input_type(dtype), addresses[extra].load(dtype))
            },
            |axis| format!("coordinate{}", coordinate_axis(rank, usize::from(axis))),
            |axis| format!("extent{}", coordinate_axis(rank, usize::from(axis))),
        );
        types.push(value_type);
        line(
            &mut source,
            format_args!(
                "    {} v{index} = {expression};",
                scalar_codegen::type_name(value_type)
            ),
        );
    }
    let value = epilogue.validated().program().outputs[0].1;
    line(&mut source, format_args!("    return v{value};"));
    line(&mut source, format_args!("}}"));
    validate_address_structure(&source, &addresses)
        .then_some(source)
        .ok_or(SourceError::AddressStructure)
}

pub(super) fn generate_source(epilogue: &PreparedEpilogue) -> Result<String, SourceError> {
    generate_source_from_header(crate::map_codegen::SHARED_HEADER, epilogue)
}

fn generate_source_from_header(
    shared_header: &str,
    epilogue: &PreparedEpilogue,
) -> Result<String, SourceError> {
    if shared_header.matches(IDENTITY_HELPER).count() != 1 {
        return Err(SourceError::IdentityHelperCount);
    }
    let helper = generate_helper(epilogue)?;
    let header = shared_header.replacen(IDENTITY_HELPER, &helper, 1);
    Ok(format!(
        "{header}\n{}",
        concat!(
            include_str!("kernels.metal"),
            "\n",
            include_str!("steel_gemm_loader.metal"),
            "\n",
            include_str!("steel_gemm_mma.metal"),
            "\n",
            include_str!("matmul.metal"),
            "\n",
            include_str!("quant_matmul.metal"),
            "\n",
            include_str!("gather_matmul.metal"),
            "\n",
            include_str!("sdpa.metal"),
            "\n",
            include_str!("sdpa_vector.metal")
        )
    ))
}

fn coordinate_axis(rank: usize, axis: usize) -> usize {
    3_usize.saturating_sub(rank) + axis
}

fn validate_address_structure(source: &str, addresses: &[AddressFragment]) -> bool {
    let address_lines = source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("ulong epilogue_address"))
        .collect::<Vec<_>>();
    address_lines.len() == addresses.len()
        && address_lines
            .iter()
            .zip(addresses)
            .all(|(actual, expected)| *actual == expected.declaration())
        && source.lines().map(str::trim).all(|line| {
            if !line.contains("epilogue_address") {
                return true;
            }
            addresses.iter().any(|address| {
                line == address.declaration()
                    || [DType::F32, DType::F16, DType::BF16, DType::U32]
                        .into_iter()
                        .any(|dtype| line.contains(&address.load(dtype)))
            })
        })
}

const fn input_type(dtype: DType) -> ValueType {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => ValueType::F32,
        DType::U32 => ValueType::U32,
        DType::I32 => unreachable!(),
    }
}

pub(super) const fn dtype_key(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::I32 => 3,
        DType::U32 => 4,
    }
}

fn line(source: &mut String, arguments: fmt::Arguments<'_>) {
    let _ = source.write_fmt(arguments);
    source.push('\n');
}

#[cfg(test)]
mod tests {
    use forja_core::program::{
        BinOp, EpilogueSignature, Inst, Program, ProgramKind, prepare_epilogue,
    };

    use super::*;

    fn bias_epilogue() -> std::sync::Arc<PreparedEpilogue> {
        prepare_epilogue(
            Program {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Input(0),
                    Inst::Input(1),
                    Inst::Binary(BinOp::Add, 0, 1),
                ],
                outputs: vec![(0, 2)],
            }
            .validate()
            .unwrap(),
            EpilogueSignature::new(1, 2, vec![DType::F16], DType::BF16),
        )
        .unwrap()
    }

    #[test]
    fn helper_addresses_depend_only_on_coordinates_and_layouts() {
        let source = generate_helper(&bias_epilogue()).unwrap();
        let address = source
            .lines()
            .find(|line| line.contains("ulong epilogue_address0"))
            .unwrap();
        assert!(address.contains("coordinate2"));
        assert!(
            !address
                .as_bytes()
                .windows(2)
                .any(|pair| { pair[0] == b'v' && pair[1].is_ascii_digit() })
        );
        assert!(source.contains("return v2;"));
    }

    #[test]
    fn pipeline_keys_canonicalize_function_constants() {
        let source = SourceKey::new(FamilySourceTag::Dense, &bias_epilogue());
        let first = PipelineKey::new(source.clone(), "gemv", vec![(2, 1), (0, 0)]);
        let second = PipelineKey::new(source, "gemv", vec![(0, 0), (2, 1)]);
        assert_eq!(first, second);
    }

    #[test]
    fn source_keys_separate_kernel_families() {
        let epilogue = bias_epilogue();
        assert_ne!(
            SourceKey::new(FamilySourceTag::Dense, &epilogue),
            SourceKey::new(FamilySourceTag::Quant, &epilogue)
        );
    }

    #[test]
    fn generated_addresses_match_typed_fragments() {
        let epilogue = bias_epilogue();
        let source = generate_helper(&epilogue).unwrap();
        let addresses = vec![AddressFragment::new(0, 2)];
        assert!(validate_address_structure(&source, &addresses));

        let mutated = source.replace("ulong(coordinate2)", "ulong(v0)");
        assert!(!validate_address_structure(&mutated, &addresses));
    }

    #[test]
    fn reformatted_identity_helper_is_rejected() {
        let reformatted =
            IDENTITY_HELPER.replace("    return accumulator0;", "  return accumulator0;");
        let header = crate::map_codegen::SHARED_HEADER.replacen(IDENTITY_HELPER, &reformatted, 1);
        assert_eq!(
            generate_source_from_header(&header, &bias_epilogue()),
            Err(SourceError::IdentityHelperCount)
        );
    }

    #[test]
    fn dense_splices_are_after_reductions_and_inside_edges() {
        let source = generate_source(&bias_epilogue()).unwrap();
        for kernel in ["kernel void gemv(", "kernel void gemv_transposed("] {
            let start = source.find(kernel).unwrap();
            let body = &source[start..];
            let next = body[1..]
                .find("kernel void ")
                .map_or(body.len(), |offset| offset + 1);
            let body = &body[..next];
            let call = body.rfind("forja_epilogue(").unwrap();
            let reduction = body.rfind("simd_sum(").unwrap();
            let edge = body[..call].rfind("if (column < params.n)").unwrap();
            let store = body[call..].find("store_float(").unwrap() + call;
            assert!(reduction < edge && edge < call && call < store);
        }
    }

    #[test]
    fn steel_splice_is_after_barriers_and_inside_edges() {
        let source = generate_source(&bias_epilogue());
        let start = source.find("void steel_gemm(").unwrap();
        let body =
            &source[start..source[start..].find("#define DEFINE_STEEL_GEMM").unwrap() + start];
        let call = body.rfind("forja_epilogue(").unwrap();
        let barrier = body.rfind("threadgroup_barrier(").unwrap();
        let edge = body[..call]
            .rfind("if (output_row < params.m && output_column < params.n)")
            .unwrap();
        let store = body[call..].find("store_float(").unwrap() + call;
        assert!(barrier < edge && edge < call && call < store);
    }

    #[test]
    fn quant_splices_follow_collectives_and_edge_checks() {
        let source = generate_source(&bias_epilogue());
        for (kernel, end_marker, expected_calls, collective, edge) in [
            (
                "kernel void quantized_gemv(",
                "kernel void quantized_gemv_q8_fast(",
                2,
                "quantized_q",
                "if (column < params.columns)",
            ),
            (
                "kernel void quantized_gemv_q8_fast(",
                "kernel void quantized_router_q8(",
                1,
                "simd_sum(",
                "if (column < params.columns)",
            ),
        ] {
            let start = source.find(kernel).unwrap();
            let body = &source[start..source[start..].find(end_marker).unwrap() + start];
            assert_eq!(body.matches("forja_epilogue(").count(), expected_calls);
            let mut end = 0;
            while let Some(offset) = body[end..].find("forja_epilogue(") {
                let call = end + offset;
                let before = &body[..call];
                assert!(before.rfind(collective).unwrap() < before.rfind(edge).unwrap());
                assert!(body[call..].find("store_float(").is_some());
                end = call + 1;
            }
        }
    }

    #[test]
    fn quant_gemm_splices_follow_collectives_and_edge_checks() {
        let source = generate_source(&bias_epilogue());
        for (kernel, end_marker, collective, edge) in [
            (
                "kernel void quantized_gemm_small_m(",
                "template <\n    typename T, uint block_inner",
                "simd_sum(",
                "if (output_row < params.rows)",
            ),
            (
                "void quantized_gemm_tiled_impl(",
                "#define quantized_gemm_tiled_kernel",
                "threadgroup_barrier(",
                "if (output_row < params.rows && output_column < params.columns)",
            ),
        ] {
            let start = source.find(kernel).unwrap();
            let body = &source[start..source[start..].find(end_marker).unwrap() + start];
            let call = body.rfind("forja_epilogue(").unwrap();
            let before = &body[..call];
            assert!(before.rfind(collective).unwrap() < before.rfind(edge).unwrap());
            assert!(body[call..].find("store_float(").is_some());
        }
    }

    #[test]
    fn quant_source_addresses_ignore_program_values() {
        let source = generate_source(&bias_epilogue());
        for address in source
            .lines()
            .filter(|line| line.contains("ulong epilogue_address"))
        {
            assert!(
                !address
                    .as_bytes()
                    .windows(2)
                    .any(|pair| pair[0] == b'v' && pair[1].is_ascii_digit())
            );
        }
    }
}
