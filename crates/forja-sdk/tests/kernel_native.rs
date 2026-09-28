#![cfg(feature = "native")]

//! Native differential coverage for Rust-syntax kernels.

use std::error::Error;

use forja_core::DType as CoreDType;
use forja_cpu::{CpuBackend, interpreter::Input};
use forja_sdk::{
    DType, Tensor,
    kernel::Kernel,
    program::{Ctx, Program},
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
