//! Differential coverage for gathered affine-quantized matrix multiplication.

use forja_core::{
    Backend, BackendDispatchData, BackendError, CommandList, DType, Op, Slice, Submission,
    program::{
        BinOp, EpilogueSignature, Inst, PreparedEpilogue, Program, ProgramKind, UnOp,
        prepare_epilogue,
    },
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;
use forja_testing::{
    DeterministicValues, TensorSpec, assert_backends_agree, assert_outputs_agree,
    generated_tensor_bytes,
};

#[derive(Clone, Copy)]
struct GroupedCase {
    rows: u32,
    routes: u32,
    experts: u32,
    inner: u32,
    columns: u32,
}

#[test]
#[ignore = "manual decode routing kernel comparison"]
fn metal_decode_batch_routing_microbenchmark() {
    let backend = MetalBackend::new().unwrap();
    for (site, inner, columns, paired) in
        [("gate-up", 2_048, 768, true), ("down", 768, 2_048, false)]
    {
        for rows in [2, 4, 8] {
            let case = GroupedCase {
                rows,
                routes: 8,
                experts: 128,
                inner,
                columns,
            };
            let specs = grouped_specs(case, paired);
            let tensors = specs
                .iter()
                .map(|spec| {
                    backend
                        .alloc(spec.dtype(), spec.allocation_shape())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let indices = tensors.last().unwrap();
            backend
                .write(indices, &vec![0; usize::try_from(rows * 8 * 4).unwrap()])
                .unwrap();
            let output = backend.alloc(DType::F32, &[rows, 8, columns]).unwrap();
            let route = || routing_benchmark_commands(&backend, &tensors, &output, paired, true);
            let grouped = || routing_benchmark_commands(&backend, &tensors, &output, paired, false);
            let (route_time, grouped_time) =
                crate::common::comparative_gpu_times(&backend, route, grouped);
            eprintln!(
                "routing-batch site={site} rows={rows} route_us={} grouped_us={} ratio={:.4}",
                route_time.as_secs_f64() * 1.0e6,
                grouped_time.as_secs_f64() * 1.0e6,
                grouped_time.as_secs_f64() / route_time.as_secs_f64(),
            );
        }
    }
}

fn routing_benchmark_commands(
    backend: &MetalBackend,
    tensors: &[forja_core::Tensor],
    output: &forja_core::Tensor,
    paired: bool,
    route: bool,
) -> CommandList {
    let op = Op::GatherQuantMatmul {
        bits: 4,
        group_size: 64,
    };
    let inputs = tensor_refs(tensors);
    let mut commands = CommandList::new();
    if paired {
        commands
            .dispatch_epilogue(op, &inputs, output, &silu_mul_epilogue(), &[])
            .unwrap();
    } else {
        commands.dispatch(op, &inputs, output).unwrap();
    }
    let name = if route {
        "gather-quant-matmul.route-gemv"
    } else if paired {
        "gather-quant-matmul.grouped-pair-small-m"
    } else {
        "gather-quant-matmul.grouped-small-m"
    };
    let variant = backend
        .validate_variant_for_op(op, &inputs, &[output], name)
        .unwrap();
    assert!(commands.set_last_backend_data(BackendDispatchData::new(variant)));
    commands
}

#[test]
fn metal_grouped_small_m_variants_match_cpu_at_edges() {
    let mut values = DeterministicValues::new(0x6a09_e667_f3bc_c909);
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for case in [
        GroupedCase {
            rows: 1,
            routes: 64,
            experts: 9,
            inner: 64,
            columns: 1,
        },
        GroupedCase {
            rows: 7,
            routes: 10,
            experts: 9,
            inner: 128,
            columns: 7,
        },
        GroupedCase {
            rows: 33,
            routes: 2,
            experts: 9,
            inner: 64,
            columns: 33,
        },
        GroupedCase {
            rows: 64,
            routes: 1,
            experts: 9,
            inner: 192,
            columns: 65,
        },
        GroupedCase {
            rows: 7,
            routes: 73,
            experts: 9,
            inner: 64,
            columns: 33,
        },
        GroupedCase {
            rows: 4_097,
            routes: 1,
            experts: 9,
            inner: 64,
            columns: 1,
        },
        GroupedCase {
            rows: 1,
            routes: 2_049,
            experts: 9,
            inner: 64,
            columns: 1,
        },
    ] {
        assert_grouped_case(
            &cpu,
            &metal,
            case,
            false,
            &mut values,
            "gather-quant-matmul.grouped-small-m",
        );
        assert_grouped_case(
            &cpu,
            &metal,
            case,
            true,
            &mut values,
            "gather-quant-matmul.grouped-pair-small-m",
        );
    }
}

#[test]
fn vector_gather_raw_and_paired_match_cpu_at_decode_shapes() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let mut values = DeterministicValues::new(0x6a09_e667_f3bc_c909);
    for inner in [64, 768, 2048] {
        for rows in [1, 2, 4, 7, 8] {
            let case = GroupedCase {
                rows,
                routes: 8,
                experts: 9,
                inner,
                columns: 33,
            };
            for variant in [
                "gather-quant-matmul.vector8",
                "gather-quant-matmul.vector16",
            ] {
                for paired in [false, true] {
                    assert_grouped_case(&cpu, &metal, case, paired, &mut values, variant);
                }
            }
        }
    }
}

#[test]
fn vector_gather_weighted_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let epilogue = prepare_epilogue(
        Program {
            kind: ProgramKind::Map,
            insts: vec![Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap(),
        EpilogueSignature::new(1, 2, vec![], DType::F32),
    )
    .unwrap();
    let mut values = DeterministicValues::new(0x6a09_e667_f3bc_c909);
    for inner in [64, 768, 2048] {
        for rows in [1, 7] {
            let case = GroupedCase {
                rows,
                routes: 8,
                experts: 9,
                inner,
                columns: 33,
            };
            let mut specs = grouped_specs(case, false);
            specs[0] = TensorSpec::contiguous(DType::F32, &[rows, 8, inner]);
            specs.push(TensorSpec::contiguous(DType::F32, &[rows, 8]));
            let bytes = specs
                .iter()
                .map(|spec| generated_tensor_bytes(spec, &mut values).unwrap())
                .collect::<Vec<_>>();
            let cpu_inputs = allocate_inputs(&cpu, &specs, &bytes);
            let metal_inputs = allocate_inputs(&metal, &specs, &bytes);
            let cpu_output = cpu.alloc(DType::F32, &[rows, 33]).unwrap();
            let metal_output = metal.alloc(DType::F32, &[rows, 33]).unwrap();
            let op = Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            };
            submit_grouped(&cpu, op, &cpu_inputs, &cpu_output, Some(&epilogue), None);
            for name in [
                "gather-quant-matmul.vector8",
                "gather-quant-matmul.vector16",
            ] {
                let refs = tensor_refs(&metal_inputs);
                let variant = metal
                    .validate_variant_for_op(op, &refs, &[&metal_output], name)
                    .unwrap();
                submit_grouped(
                    &metal,
                    op,
                    &metal_inputs,
                    &metal_output,
                    Some(&epilogue),
                    Some(BackendDispatchData::new(variant)),
                );
                assert_outputs_agree(
                    DType::F32,
                    &cpu.read(&cpu_output).unwrap(),
                    &metal.read(&metal_output).unwrap(),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn metal_grouped_tile_heights_match_bits() {
    let metal = MetalBackend::new().unwrap();
    let mut values = DeterministicValues::new(0xbb67_ae85_84ca_a73b);
    for (rows, routes, expert_routes) in [(33, 2, 66), (64, 1, 64)]
        .into_iter()
        .chain((8..=17).map(|count| (64, 1, count)))
        .chain((72..=81).map(|rows| (rows, 1, rows)))
    {
        let case = GroupedCase {
            rows,
            routes,
            experts: 9,
            inner: 128,
            columns: 33,
        };
        for paired in [false, true] {
            let specs = grouped_specs(case, paired);
            let mut bytes = specs
                .iter()
                .map(|spec| generated_tensor_bytes(spec, &mut values).unwrap())
                .collect::<Vec<_>>();
            *bytes.last_mut().unwrap() = (0..rows * routes)
                .flat_map(|route| {
                    let expert = if route < expert_routes { 4_u32 } else { 5 };
                    expert.to_le_bytes()
                })
                .collect();
            let inputs = allocate_inputs(&metal, &specs, &bytes);
            let op = Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            };
            let epilogue = paired.then(silu_mul_epilogue);
            let mut outputs = Vec::new();
            let mut variants = vec![
                "gather-quant-matmul.grouped",
                "gather-quant-matmul.grouped-32x32-2x2",
                "gather-quant-matmul.grouped-32x32-2x2-blocked",
            ];
            if paired {
                variants.push("gather-quant-matmul.grouped-pair-32x32-2x2-blocked");
            }
            for name in variants {
                let output = metal
                    .alloc(DType::F32, &[case.rows, case.routes, case.columns])
                    .unwrap();
                let refs = tensor_refs(&inputs);
                let variant = metal
                    .validate_variant_for_op(op, &refs, &[&output], name)
                    .unwrap();
                submit_grouped(
                    &metal,
                    op,
                    &inputs,
                    &output,
                    epilogue.as_ref(),
                    Some(BackendDispatchData::new(variant)),
                );
                outputs.push(metal.read(&output).unwrap());
            }
            assert_eq!(outputs[0], outputs[1]);
            assert_eq!(
                outputs[0], outputs[2],
                "blocked: rows={rows}, routes={routes}, expert_routes={expert_routes}, paired={paired}"
            );
            if paired {
                assert_eq!(
                    outputs[0], outputs[3],
                    "fused: rows={rows}, routes={routes}, expert_routes={expert_routes}"
                );
            }
        }
    }
}

fn assert_grouped_case(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    case: GroupedCase,
    paired: bool,
    values: &mut DeterministicValues,
    variant: &str,
) {
    let specs = grouped_specs(case, paired);
    let bytes = specs
        .iter()
        .map(|spec| generated_tensor_bytes(spec, values).unwrap())
        .collect::<Vec<_>>();
    let cpu_inputs = allocate_inputs(cpu, &specs, &bytes);
    let metal_inputs = allocate_inputs(metal, &specs, &bytes);
    let shape = [case.rows, case.routes, case.columns];
    let cpu_output = cpu.alloc(DType::F32, &shape).unwrap();
    let metal_output = metal.alloc(DType::F32, &shape).unwrap();
    let op = Op::GatherQuantMatmul {
        bits: 4,
        group_size: 64,
    };
    let epilogue = paired.then(silu_mul_epilogue);
    submit_grouped(cpu, op, &cpu_inputs, &cpu_output, epilogue.as_ref(), None);
    let inputs = tensor_refs(&metal_inputs);
    let variant = metal
        .validate_variant_for_op(op, &inputs, &[&metal_output], variant)
        .unwrap();
    submit_grouped(
        metal,
        op,
        &metal_inputs,
        &metal_output,
        epilogue.as_ref(),
        Some(BackendDispatchData::new(variant)),
    );
    assert_outputs_agree(
        DType::F32,
        &cpu.read(&cpu_output).unwrap(),
        &metal.read(&metal_output).unwrap(),
    )
    .unwrap();
}

fn grouped_specs(case: GroupedCase, paired: bool) -> Vec<TensorSpec> {
    let GroupedCase {
        rows,
        routes,
        experts,
        inner,
        columns,
    } = case;
    let mut specs = vec![
        TensorSpec::contiguous(DType::F32, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
    ];
    if paired {
        specs.extend([
            TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]),
            TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
            TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]),
        ]);
    }
    let total = rows * routes;
    let mut selected = (1..=7)
        .flat_map(|count| std::iter::repeat_n(count - 1, usize::try_from(count).unwrap()))
        .collect::<Vec<u32>>();
    selected.resize(usize::try_from(total).unwrap(), experts - 1);
    specs.push(TensorSpec::initialized(
        DType::U32,
        &[rows, routes],
        selected.into_iter().flat_map(u32::to_le_bytes).collect(),
    ));
    specs
}

fn silu_mul_epilogue() -> std::sync::Arc<PreparedEpilogue> {
    prepare_epilogue(
        Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Input(1),
                Inst::Unary(UnOp::Neg, 0),
                Inst::Unary(UnOp::Exp, 2),
                Inst::Const(1.0),
                Inst::Binary(BinOp::Add, 4, 3),
                Inst::Binary(BinOp::Div, 0, 5),
                Inst::Binary(BinOp::Mul, 6, 1),
            ],
            outputs: vec![(0, 7)],
        }
        .validate()
        .unwrap(),
        EpilogueSignature::new(2, 3, vec![], DType::F32),
    )
    .unwrap()
}

fn submit_grouped<B: Backend>(
    backend: &B,
    op: Op,
    inputs: &[forja_core::Tensor],
    output: &forja_core::Tensor,
    epilogue: Option<&std::sync::Arc<PreparedEpilogue>>,
    variant: Option<BackendDispatchData>,
) {
    let mut commands = CommandList::new();
    let inputs = tensor_refs(inputs);
    if let Some(epilogue) = epilogue {
        commands
            .dispatch_epilogue(op, &inputs, output, epilogue, &[])
            .unwrap();
    } else {
        commands.dispatch(op, &inputs, output).unwrap();
    }
    if let Some(variant) = variant {
        assert!(commands.set_last_backend_data(variant));
    }
    backend.submit(commands).unwrap().wait().unwrap();
}

fn allocate_inputs<B: Backend>(
    backend: &B,
    specs: &[TensorSpec],
    bytes: &[Vec<u8>],
) -> Vec<forja_core::Tensor> {
    specs
        .iter()
        .zip(bytes)
        .map(|(spec, bytes)| {
            let tensor = backend
                .alloc(spec.dtype(), spec.allocation_shape())
                .unwrap();
            backend.write(&tensor, bytes).unwrap();
            tensor
        })
        .collect()
}

fn tensor_refs(tensors: &[forja_core::Tensor]) -> Vec<&forja_core::Tensor> {
    tensors.iter().collect()
}

#[test]
fn expert_128_projection_shapes_match_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [1, 8, 128, 2048, 768], dtype);
        assert_case(&cpu, &metal, [1, 8, 128, 768, 2048], dtype);
    }
}

#[test]
fn small_row_gathered_quantized_matmul_matches_cpu_with_duplicates() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    assert_case(&cpu, &metal, [7, 3, 5, 64, 33], DType::BF16);
    assert_case(&cpu, &metal, [33, 3, 5, 64, 33], DType::BF16);
}

#[test]
fn grouped_1024_by_128_projection_matches_cpu() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        assert_case(&cpu, &metal, [16, 8, 128, 1024, 128], dtype);
    }
}

#[test]
fn grouped_quantized_ops_match_cpu_for_strided_f32_inputs() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let rows = 16;
    let inner = 64;
    let row_slice = Slice::new(0, rows, 1).unwrap();
    let activations = [
        TensorSpec::sliced_broadcast(
            DType::F32,
            &[rows, 4],
            &[row_slice, Slice::new(0, 1, 1).unwrap()],
            &[rows, inner],
        ),
        TensorSpec::sliced(
            DType::F32,
            &[rows, inner * 2],
            &[row_slice, Slice::new(0, inner, 2).unwrap()],
        ),
    ];
    for activation in activations {
        assert_grouped_strided_case(&cpu, &metal, activation);
    }
}

fn assert_grouped_strided_case(cpu: &CpuBackend, metal: &MetalBackend, activation: TensorSpec) {
    let rows = 16;
    let routes = 4;
    let experts = 8;
    let inner = 64;
    let columns = 33;
    let selected = (0..rows * routes)
        .map(|route| route % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let packed = TensorSpec::contiguous(DType::U32, &[experts, columns, inner / 8]);
    let scales = TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]);
    let biases = TensorSpec::contiguous(DType::F16, &[experts, columns, inner / 64]);
    let indices = TensorSpec::initialized(DType::U32, &[rows, routes], selected);
    let output = TensorSpec::contiguous(DType::F32, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmul {
            bits: 4,
            group_size: 64,
        },
        &[activation, packed, scales, biases, indices],
        &output,
    )
    .unwrap();
}

#[test]
fn grouped_matmul_zeros_gpu_written_out_of_range_routes() {
    let metal = MetalBackend::new().unwrap();
    let input = metal.alloc(DType::F32, &[33, 64]).unwrap();
    let packed = metal.alloc(DType::U32, &[3, 2, 8]).unwrap();
    let scales = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let biases = metal.alloc(DType::F16, &[3, 2, 1]).unwrap();
    let source = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let mut selected = vec![0_u32; 66];
    selected[..2].copy_from_slice(&[99, 7]);
    metal
        .write(
            &source,
            &selected
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let indices = metal.alloc(DType::U32, &[33, 2]).unwrap();
    let output = metal.alloc(DType::F32, &[33, 2, 2]).unwrap();
    metal.write(&output, &[0xa5; 33 * 2 * 2 * 4]).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    commands
        .dispatch(
            Op::GatherQuantMatmul {
                bits: 4,
                group_size: 64,
            },
            &[&input, &packed, &scales, &biases, &indices],
            &output,
        )
        .unwrap();
    assert_error_and_zero_output(
        &metal,
        metal.submit(commands).unwrap(),
        &output,
        33 * 2 * 2 * 4,
        7,
        "gather-quant-matmul",
    );
}

#[test]
fn decode_matmul_zeros_gpu_written_out_of_range_routes() {
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let metal = MetalBackend::new().unwrap();
        let input = metal.alloc(dtype, &[8, 768]).unwrap();
        let packed = metal.alloc(DType::U32, &[128, 2048, 96]).unwrap();
        let scales = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let biases = metal.alloc(DType::F16, &[128, 2048, 12]).unwrap();
        let indices = copied_invalid_indices(&metal, &[8, 1]);
        let output = metal.alloc(dtype, &[8, 1, 2048]).unwrap();
        let output_bytes = usize::try_from(8 * 2048 * dtype.byte_size()).unwrap();
        metal.write(&output, &vec![0xa5; output_bytes]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &[&input, &packed, &scales, &biases, &indices],
                &output,
            )
            .unwrap();
        assert_error_and_zero_output(
            &metal,
            metal.submit(commands).unwrap(),
            &output,
            output_bytes,
            129,
            "gather-quant-matmul",
        );
    }
}

fn copied_invalid_indices(metal: &MetalBackend, shape: &[u32]) -> forja_core::Tensor {
    let source = metal.alloc(DType::U32, shape).unwrap();
    let indices = metal.alloc(DType::U32, shape).unwrap();
    let mut selected = vec![0_u32; usize::try_from(shape.iter().product::<u32>()).unwrap()];
    selected[0] = 129;
    metal
        .write(
            &source,
            &selected
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&source], &indices).unwrap();
    metal.submit(commands).unwrap().wait().unwrap();
    indices
}

fn assert_error_and_zero_output(
    metal: &MetalBackend,
    submission: impl Submission,
    output: &forja_core::Tensor,
    len: usize,
    index: u32,
    operation: &'static str,
) {
    assert!(matches!(
        submission.wait(),
        Err(BackendError::DeviceErrorFlag {
            operation: actual_operation,
            flag: 1,
            value,
            ..
        }) if actual_operation == operation && value == index
    ));
    drop(submission);
    assert_eq!(metal.read(output).unwrap(), vec![0; len]);
}

fn assert_case(cpu: &CpuBackend, metal: &MetalBackend, shape: [u32; 5], dtype: DType) {
    let [rows, routes, experts, inner, columns] = shape;
    let bits = 4;
    let group_size = 64;
    let packed_width = inner * u32::from(bits) / 32;
    let groups = inner / group_size;
    let selected = (0..rows * routes)
        .map(|slot| (slot / 2) % experts)
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let inputs = [
        TensorSpec::contiguous(dtype, &[rows, inner]),
        TensorSpec::contiguous(DType::U32, &[experts, columns, packed_width]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::contiguous(DType::F16, &[experts, columns, groups]),
        TensorSpec::initialized(DType::U32, &[rows, routes], selected),
    ];
    let output = TensorSpec::contiguous(dtype, &[rows, routes, columns]);
    assert_backends_agree(
        cpu,
        metal,
        Op::GatherQuantMatmul { bits, group_size },
        &inputs,
        &output,
    )
    .unwrap();
}
