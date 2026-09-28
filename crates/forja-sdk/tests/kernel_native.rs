#![cfg(feature = "native")]

//! Native differential coverage for Rust-syntax kernels.

use std::error::Error;

use forja_core::{DType as CoreDType, Slice};
use forja_cpu::{CpuBackend, interpreter::Input};
use forja_sdk::{
    DType, Tensor,
    kernel::Kernel,
    program::{Ctx, Program, RowCtx},
};
use forja_testing::{
    DeterministicValues, TensorSpec, assert_f32_values_agree, assert_program_backends_agree,
    program::ProgramCase,
};

#[forja_sdk::kernel(map)]
fn macro_silu_mul(
    gate: forja_sdk::kernel::Elem,
    up: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    gate * gate.sigmoid() * up
}

#[forja_sdk::kernel(map)]
fn macro_map_norm(
    x: forja_sdk::kernel::Elem,
    weight: forja_sdk::kernel::Elem,
    epsilon: f32,
) -> forja_sdk::kernel::Elem {
    x * (x * x + epsilon).rsqrt() * weight
}

#[forja_sdk::kernel(map)]
fn macro_multi(
    x: forja_sdk::kernel::Elem,
    y: forja_sdk::kernel::Elem,
) -> (
    forja_sdk::kernel::Elem,
    forja_sdk::kernel::Elem,
    forja_sdk::kernel::Elem,
) {
    (x + y, x - y, x * y)
}

#[forja_sdk::kernel(map)]
fn macro_cached_scalar(x: forja_sdk::kernel::Elem, position: f32) -> forja_sdk::kernel::Elem {
    x + position
}

#[forja_sdk::kernel(row)]
fn macro_typed_row(
    x: forja_sdk::kernel::Row,
    ids: forja_sdk::kernel::Row<u32>,
) -> forja_sdk::kernel::Row {
    let lane = forja_sdk::kernel::index(-1);
    let width = x.len().min(forja_sdk::kernel::extent(-1));
    let wrapped = ids
        .wrapping_add(lane)
        .wrapping_mul(3)
        .wrapping_sub(1)
        .min(width)
        .max(0);
    let choose = (x >= 0.0 && lane < width) || !(ids != 0);
    let selected = if choose {
        wrapped
    } else if ids > lane {
        choose as u32
    } else {
        x as u32
    };
    let value = if choose {
        x.maximum(selected as f32)
    } else {
        x.minimum(selected as f32)
    };
    value + value.row_mean()
}

#[forja_sdk::kernel(map)]
fn macro_bad_axis(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x + forja_sdk::kernel::index(2) as f32
}

#[forja_sdk::kernel(helper)]
fn helper_square(x: f32) -> f32 {
    x * x
}

#[forja_sdk::kernel(map)]
fn macro_helper_square(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    helper_square(x)
}

#[forja_sdk::kernel(helper)]
fn helper_sixteen(mut_x: f32) -> f32 {
    let x = mut_x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    let x = x + 1.0;
    x + 1.0
}

#[forja_sdk::kernel(helper)]
fn helper_too_wide(x: f32) -> f32 {
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    let x = helper_sixteen(x);
    helper_sixteen(x)
}

#[forja_sdk::kernel(map)]
fn macro_helper_too_wide(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    helper_too_wide(x)
}

struct TestInput {
    values: Vec<f32>,
    shape: Vec<u32>,
    logical: Vec<f32>,
}

fn random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut values = DeterministicValues::new(seed);
    (0..count).map(|_| values.next_f32()).collect()
}

fn differential(
    shape: &[u32],
    inputs: &[TestInput],
    macro_kernel: &Kernel,
    hand_program: &Program,
    check_identity: bool,
    run_macro: impl Fn(&[Tensor<f32>]) -> forja_sdk::Result<Vec<Tensor<f32>>>,
) -> Result<(), Box<dyn Error>> {
    let rank = u8::try_from(shape.len())?;
    let input_dtypes = vec![DType::F32; inputs.len()];
    let output_count = macro_kernel.validated_program().output_count();
    let output_dtypes = vec![DType::F32; output_count];
    let hand_kernel = Kernel::new(hand_program, rank, &input_dtypes, &output_dtypes)?;
    if check_identity {
        assert_eq!(
            macro_kernel.validated_program().content_hash(),
            hand_kernel.validated_program().content_hash()
        );
    }

    let input_specs = inputs
        .iter()
        .map(|input| {
            if input.shape == shape {
                TensorSpec::contiguous(CoreDType::F32, &input.shape)
            } else {
                TensorSpec::broadcast(CoreDType::F32, &input.shape, shape)
            }
        })
        .collect();
    let output_specs = (0..output_count)
        .map(|_| TensorSpec::contiguous(CoreDType::F32, shape))
        .collect();
    let case = ProgramCase::new(
        macro_kernel.validated_program().program().clone(),
        shape.to_vec(),
        input_specs,
        output_specs,
    );
    let cpu = CpuBackend::new();
    let report = assert_program_backends_agree(&cpu, &cpu, &case)?;
    assert!(report.total_elements > 0);

    let interpreter_inputs = inputs
        .iter()
        .map(|input| Input::F32(input.logical.as_slice()))
        .collect::<Vec<_>>();
    let macro_reference = forja_cpu::interpreter::interpret(
        macro_kernel.validated_program(),
        shape,
        &interpreter_inputs,
    )?;
    let hand_reference = forja_cpu::interpreter::interpret(
        hand_kernel.validated_program(),
        shape,
        &interpreter_inputs,
    )?;
    for (expected, actual) in hand_reference.iter().zip(&macro_reference) {
        assert_f32_values_agree(expected, actual)?;
    }

    let tensors = inputs
        .iter()
        .map(|input| Tensor::from_slice(&input.values, &input.shape))
        .collect::<forja_sdk::Result<Vec<_>>>()?;
    let macro_outputs = run_macro(&tensors)?;
    let broadcast = tensors
        .iter()
        .skip(1)
        .map(|tensor| {
            if tensor.shape() == shape {
                Ok(None)
            } else {
                tensor.broadcast_as(shape).map(Some)
            }
        })
        .collect::<forja_sdk::Result<Vec<_>>>()?;
    let additional = tensors
        .iter()
        .skip(1)
        .zip(&broadcast)
        .map(|(tensor, view)| view.as_ref().unwrap_or(tensor))
        .collect::<Vec<_>>();
    let hand_outputs = tensors[0].run_program(hand_program, &additional)?;
    for (expected, actual) in hand_outputs.iter().zip(&macro_outputs) {
        assert_f32_values_agree(&expected.to_vec()?, &actual.to_vec()?)?;
    }
    Ok(())
}

#[test]
fn silu_mul_matches_builder_and_interpreter() -> Result<(), Box<dyn Error>> {
    let shape = [7, 1024];
    let inputs = [input(&shape, 0x1234), input(&shape, 0x5678)];
    let macro_kernel =
        macro_silu_mul_program(shape.len(), &[DType::F32, DType::F32], &[DType::F32])?;
    let hand = Ctx::new();
    let gate = hand.input(0);
    hand.output(0, gate * gate.sigmoid() * hand.input(1));

    differential(
        &shape,
        &inputs,
        &macro_kernel,
        &hand.finish(),
        true,
        |input| Ok(vec![macro_silu_mul(&input[0], &input[1])?]),
    )
}

#[test]
fn map_norm_broadcast_matches_builder_and_interpreter() -> Result<(), Box<dyn Error>> {
    let shape = [7, 1024];
    let x = input(&shape, 0x9abc);
    let weight_values = random_values(1024, 0xdef0);
    let weight = TestInput {
        logical: weight_values.repeat(7),
        values: weight_values,
        shape: vec![1024],
    };
    let epsilon = 1.0e-6;
    let macro_kernel = macro_map_norm_program(
        shape.len(),
        &[DType::F32, DType::F32],
        &[DType::F32],
        epsilon,
    )?;
    let hand = Ctx::new();
    let value = hand.input(0);
    hand.output(0, value * (value * value + epsilon).rsqrt() * hand.input(1));

    differential(
        &shape,
        &[x, weight],
        &macro_kernel,
        &hand.finish(),
        false,
        |input| Ok(vec![macro_map_norm(&input[0], &input[1], epsilon)?]),
    )
}

#[test]
fn multiple_outputs_match_builder_and_interpreter() -> Result<(), Box<dyn Error>> {
    let shape = [4097];
    let inputs = [input(&shape, 0x1357), input(&shape, 0x2468)];
    let macro_kernel = macro_multi_program(
        shape.len(),
        &[DType::F32, DType::F32],
        &[DType::F32, DType::F32, DType::F32],
    )?;
    let hand = Ctx::new();
    let x = hand.input(0);
    let y = hand.input(1);
    hand.output(0, x + y);
    hand.output(1, x - y);
    hand.output(2, x * y);

    differential(
        &shape,
        &inputs,
        &macro_kernel,
        &hand.finish(),
        false,
        |input| {
            let (sum, difference, product) = macro_multi(&input[0], &input[1])?;
            Ok(vec![sum, difference, product])
        },
    )
}

#[test]
fn macro_cache_rejects_the_seventeenth_signature() -> Result<(), Box<dyn Error>> {
    for position in 0_u8..16 {
        macro_cached_scalar_program(1, &[DType::F32], &[DType::F32], f32::from(position))?;
    }
    let error = macro_cached_scalar_program(1, &[DType::F32], &[DType::F32], 16.0)
        .err()
        .expect("the seventeenth signature must be rejected");
    assert_eq!(
        error.to_string(),
        "kernel `macro_cached_scalar`: more than 16 distinct (rank, dtype, scalar) variants; pass varying scalars as tensors"
    );
    Ok(())
}

#[test]
fn typed_row_matches_builder_and_interpreter() -> Result<(), Box<dyn Error>> {
    let (macro_kernel, hand_kernel) = typed_kernels()?;
    let cpu = CpuBackend::new();
    for kernel in [&macro_kernel, &hand_kernel] {
        let report = assert_program_backends_agree(&cpu, &cpu, &typed_case(kernel))?;
        assert!(report.reduction_steps > 0);
    }

    let shape = [7, 33];
    let count = 7 * 33;
    let floats = random_values(count, 0xfeed);
    let integers = (0..count)
        .map(|index| u32::try_from(index % 41).unwrap())
        .collect::<Vec<_>>();
    let inputs = [Input::F32(&floats), Input::U32(&integers)];
    let macro_outputs =
        forja_cpu::interpreter::interpret(macro_kernel.validated_program(), &shape, &inputs)?;
    let hand_outputs =
        forja_cpu::interpreter::interpret(hand_kernel.validated_program(), &shape, &inputs)?;
    assert_f32_values_agree(&hand_outputs[0], &macro_outputs[0])?;
    Ok(())
}

#[test]
fn axis_errors_name_the_kernel() {
    let error = macro_bad_axis_program(2, &[DType::F32], &[DType::F32])
        .err()
        .expect("the third axis must be rejected for rank two");
    assert_eq!(
        error.to_string(),
        "kernel `macro_bad_axis`: program axis is out of range"
    );
}

#[test]
fn helpers_match_builder_and_interpreter() -> Result<(), Box<dyn Error>> {
    let shape = [4097];
    let inputs = [input(&shape, 0xaaaa)];
    let macro_kernel = macro_helper_square_program(1, &[DType::F32], &[DType::F32])?;
    let hand = Ctx::new();
    let x = hand.input(0);
    hand.output(0, x * x);

    differential(
        &shape,
        &inputs,
        &macro_kernel,
        &hand.finish(),
        true,
        |input| Ok(vec![macro_helper_square(&input[0])?]),
    )
}

#[test]
fn helper_expansion_caps_name_the_kernel() {
    let instructions = macro_helper_too_wide_program(1, &[DType::F32], &[DType::F32])
        .err()
        .expect("expanded helpers must exceed the instruction cap");
    assert_eq!(
        instructions.to_string(),
        "kernel `macro_helper_too_wide`: invalid program: TooManyInstructions"
    );
}

fn typed_kernels() -> forja_sdk::Result<(Kernel, Kernel)> {
    let macro_kernel = macro_typed_row_program(2, &[DType::F32, DType::U32], &[DType::F32])?;
    let context = RowCtx::new();
    let x = context.input(0);
    let ids = context.input_u32(1);
    let lane = context.index(-1);
    let width = context.extent(-1).min(context.extent(-1));
    let three = context.constant(3.0).cast_u32();
    let one = context.constant(1.0).cast_u32();
    let zero = context.constant(0.0).cast_u32();
    let wrapped = ids
        .wrapping_add(lane)
        .wrapping_mul(three)
        .wrapping_sub(one)
        .min(width)
        .max(zero);
    let choose = x
        .ge(context.constant(0.0))
        .and(lane.lt(width))
        .or(ids.not_equal(zero).not());
    let selected = choose.select_u32(
        wrapped,
        ids.gt(lane).select_u32(choose.cast_u32(), x.cast_u32()),
    );
    let selected = selected.cast_f32();
    let value = choose.select(x.maximum(selected), x.minimum(selected));
    context.output(0, value + context.row_mean(value));
    let hand_kernel = Kernel::new(
        &context.finish(),
        2,
        &[DType::F32, DType::U32],
        &[DType::F32],
    )?;
    Ok((macro_kernel, hand_kernel))
}

fn typed_case(kernel: &Kernel) -> ProgramCase {
    let slices = [Slice::new(0, 7, 1).unwrap(), Slice::new(1, 33, 1).unwrap()];
    ProgramCase::new(
        kernel.validated_program().program().clone(),
        vec![7, 33],
        vec![
            TensorSpec::sliced(CoreDType::F32, &[7, 34], &slices),
            TensorSpec::sliced(CoreDType::U32, &[7, 34], &slices),
        ],
        vec![TensorSpec::contiguous(CoreDType::F32, &[7, 33])],
    )
}

fn input(shape: &[u32], seed: u64) -> TestInput {
    let count = shape
        .iter()
        .map(|&extent| usize::try_from(extent).unwrap())
        .product();
    let values = random_values(count, seed);
    TestInput {
        logical: values.clone(),
        values,
        shape: shape.to_vec(),
    }
}

#[cfg(all(feature = "native-metal", target_os = "macos"))]
#[test]
fn metal_macro_programs_match_interval_oracle() -> Result<(), Box<dyn Error>> {
    use forja_metal::MetalBackend;
    use forja_sdk::{NativeDevice, set_native_device};

    set_native_device(NativeDevice::Metal);
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new()?;
    let silu = macro_silu_mul_program(2, &[DType::F32, DType::F32], &[DType::F32])?;
    let norm = macro_map_norm_program(2, &[DType::F32, DType::F32], &[DType::F32], 1.0e-6)?;
    let multi = macro_multi_program(
        1,
        &[DType::F32, DType::F32],
        &[DType::F32, DType::F32, DType::F32],
    )?;
    let cases = [
        interval_case(&silu, &[7, 1024], &[[7, 1024], [7, 1024]], 1),
        interval_case(&norm, &[7, 1024], &[[7, 1024], [1, 1024]], 1),
        interval_case(&multi, &[4097], &[[4097, 1], [4097, 1]], 3),
    ];
    for case in &cases {
        assert_program_backends_agree(&cpu, &metal, case)?;
    }
    let (typed_macro, typed_hand) = typed_kernels()?;
    for kernel in [&typed_macro, &typed_hand] {
        assert_program_backends_agree(&cpu, &metal, &typed_case(kernel))?;
    }
    Ok(())
}

#[cfg(all(feature = "native-metal", target_os = "macos"))]
fn interval_case(
    kernel: &Kernel,
    shape: &[u32],
    input_shapes: &[[u32; 2]],
    output_count: usize,
) -> ProgramCase {
    let inputs = input_shapes
        .iter()
        .map(|input_shape| {
            let input_shape = if input_shape[1] == 1 {
                &input_shape[..1]
            } else {
                &input_shape[..]
            };
            if input_shape == shape {
                TensorSpec::contiguous(CoreDType::F32, input_shape)
            } else {
                TensorSpec::broadcast(CoreDType::F32, input_shape, shape)
            }
        })
        .collect();
    let outputs = (0..output_count)
        .map(|_| TensorSpec::contiguous(CoreDType::F32, shape))
        .collect();
    ProgramCase::new(
        kernel.validated_program().program().clone(),
        shape.to_vec(),
        inputs,
        outputs,
    )
}
