//! Bit-exact generated and trusted f32 store rounding tests.

#[path = "common/rounding_program.rs"]
mod rounding_program;

use forja_core::{
    Backend, CommandList, DType, Op, Submission,
    program::{Inst, Program, ProgramKind, ValidatedProgram},
};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;

#[test]
fn f32_stores_round_to_nearest_even_in_every_program_context() {
    let values = midpoint_sweep();
    let minimal = Program {
        kind: ProgramKind::Map,
        insts: vec![Inst::Input(0)],
        outputs: vec![(0, 0)],
    }
    .validate()
    .unwrap();
    for dtype in [DType::F16, DType::BF16] {
        assert_program_bits(&minimal, dtype, &values, None);
        for width in [33, 1025] {
            let mut context = rounding_program::context_program();
            context.insts.push(Inst::Input(2));
            context.outputs = vec![(0, 29), (1, 64)];
            assert_program_bits(&context.validate().unwrap(), dtype, &values, Some(width));
        }
        assert_copy_bits(dtype, &values);
    }
}

fn assert_program_bits(
    program: &ValidatedProgram,
    dtype: DType,
    values: &[f32],
    context_width: Option<usize>,
) {
    let width = context_width.unwrap_or(values.len());
    let cpu = run_program(
        &CpuBackend::new(),
        program,
        dtype,
        values,
        width,
        context_width.is_some(),
    );
    let metal = run_program(
        &MetalBackend::new().unwrap(),
        program,
        dtype,
        values,
        width,
        context_width.is_some(),
    );
    assert_eq!(
        metal, cpu,
        "{dtype:?} program stores differed at width {width}"
    );
}

fn run_program<B: Backend>(
    backend: &B,
    program: &ValidatedProgram,
    dtype: DType,
    values: &[f32],
    width: usize,
    context: bool,
) -> Vec<Vec<u8>> {
    let shape = [u32::try_from(width).unwrap()];
    let sweep = repeat_values(values, width);
    let sweep_bytes = sweep
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let sweep_input = backend.alloc(DType::F32, &shape).unwrap();
    backend.write(&sweep_input, &sweep_bytes).unwrap();
    let mut owned_inputs = vec![sweep_input];
    if context {
        let context_input = backend.alloc(DType::BF16, &shape).unwrap();
        backend
            .write(&context_input, &[0x80_u8, 0x3f].repeat(width))
            .unwrap();
        let indices = backend.alloc(DType::U32, &shape).unwrap();
        backend.write(&indices, &vec![0_u8; width * 4]).unwrap();
        owned_inputs = vec![context_input, indices, owned_inputs.remove(0)];
    }
    let outputs = program
        .program()
        .outputs
        .iter()
        .map(|_| backend.alloc(dtype, &shape).unwrap())
        .collect::<Vec<_>>();
    let mut commands = CommandList::new();
    commands
        .dispatch_program(
            program,
            &owned_inputs.iter().collect::<Vec<_>>(),
            &outputs.iter().collect::<Vec<_>>(),
        )
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    outputs
        .iter()
        .map(|output| backend.read(output).unwrap())
        .collect()
}

fn assert_copy_bits(dtype: DType, values: &[f32]) {
    let shape = [u32::try_from(values.len()).unwrap()];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let cpu = copy(&CpuBackend::new(), dtype, &shape, &bytes);
    let metal = copy(&MetalBackend::new().unwrap(), dtype, &shape, &bytes);
    assert_eq!(metal, cpu, "{dtype:?} trusted copy store differed");
}

fn copy<B: Backend>(backend: &B, dtype: DType, shape: &[u32], bytes: &[u8]) -> Vec<u8> {
    let input = backend.alloc(DType::F32, shape).unwrap();
    backend.write(&input, bytes).unwrap();
    let output = backend.alloc(dtype, shape).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&input], &output).unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    backend.read(&output).unwrap()
}

fn midpoint_sweep() -> Vec<f32> {
    let midpoints = [
        0x3f00_1000_u32,
        0x3f00_3000,
        0x3f1f_5000,
        0x3f1f_7000,
        0x3f80_1000,
        0x3f80_3000,
    ]
    .into_iter()
    .flat_map(|bits| {
        let midpoint = f32::from_bits(bits);
        [f32::from_bits(bits - 1), midpoint, f32::from_bits(bits + 1)]
    });
    let edges = [
        0x0000_0000_u32,
        0x32ff_ffff,
        0x3300_0000,
        0x3300_0001,
        0x3380_0000,
        0x387f_c000,
        0x387f_dfff,
        0x387f_e000,
        0x387f_e001,
        0x477f_e000,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x7f7f_0000,
        0x7f7f_7fff,
        0x7f7f_8000,
        0x7f7f_8001,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc0_0000,
    ]
    .into_iter()
    .map(f32::from_bits);
    midpoints
        .chain(edges)
        .flat_map(|value| [value, -value])
        .collect()
}

fn repeat_values(values: &[f32], len: usize) -> Vec<f32> {
    values.iter().copied().cycle().take(len).collect()
}
