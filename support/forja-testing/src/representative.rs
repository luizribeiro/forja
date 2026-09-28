//! Representative scalar programs used by backend tests.

use forja_core::program::{BinOp, Inst, Program, ProgramKind, RedOp, UnOp, ValueType};

/// Builds row-wise RMS normalization with a broadcast weight input.
#[must_use]
pub fn rms_norm(epsilon: f32) -> Program {
    let mut builder = Builder::new(ProgramKind::Row);
    let value = builder.input(0);
    let weight = builder.input(1);
    let square = builder.binary(BinOp::Mul, value, value);
    let sum = builder.reduce(RedOp::Sum, square);
    let width = builder.extent(1);
    let width = builder.cast(ValueType::F32, width);
    let mean = builder.binary(BinOp::Div, sum, width);
    let epsilon = builder.constant(epsilon);
    let stabilized = builder.binary(BinOp::Add, mean, epsilon);
    let inverse_rms = builder.unary(UnOp::Rsqrt, stabilized);
    let normalized = builder.binary(BinOp::Mul, value, inverse_rms);
    let scaled = builder.binary(BinOp::Mul, normalized, weight);
    builder.output(0, scaled);
    builder.finish()
}

/// Builds fused residual addition followed by row-wise RMS normalization.
#[must_use]
pub fn residual_rms_norm(epsilon: f32) -> Program {
    let mut builder = Builder::new(ProgramKind::Row);
    let residual = builder.input(0);
    let update = builder.input(1);
    let weight = builder.input(2);
    let value = builder.binary(BinOp::Add, residual, update);
    let square = builder.binary(BinOp::Mul, value, value);
    let sum = builder.reduce(RedOp::Sum, square);
    let width = builder.extent(1);
    let width = builder.cast(ValueType::F32, width);
    let mean = builder.binary(BinOp::Div, sum, width);
    let epsilon = builder.constant(epsilon);
    let stabilized = builder.binary(BinOp::Add, mean, epsilon);
    let inverse_rms = builder.unary(UnOp::Rsqrt, stabilized);
    let normalized = builder.binary(BinOp::Mul, value, inverse_rms);
    let scaled = builder.binary(BinOp::Mul, normalized, weight);
    builder.output(0, scaled);
    builder.finish()
}

/// Builds stable row-wise softmax.
#[must_use]
pub fn softmax() -> Program {
    let mut builder = Builder::new(ProgramKind::Row);
    let value = builder.input(0);
    let maximum = builder.reduce(RedOp::Max, value);
    let centered = builder.binary(BinOp::Sub, value, maximum);
    let exponent = builder.unary(UnOp::Exp, centered);
    let denominator = builder.reduce(RedOp::Sum, exponent);
    let probability = builder.binary(BinOp::Div, exponent, denominator);
    builder.output(0, probability);
    builder.finish()
}

/// Builds elementwise SiLU-gate multiplication.
#[must_use]
pub fn silu_mul() -> Program {
    let mut builder = Builder::new(ProgramKind::Map);
    let gate = builder.input(0);
    let up = builder.input(1);
    let sigmoid = builder.unary(UnOp::Sigmoid, gate);
    let silu = builder.binary(BinOp::Mul, gate, sigmoid);
    let activated = builder.binary(BinOp::Mul, silu, up);
    builder.output(0, activated);
    builder.finish()
}

/// Builds elementwise residual addition.
#[must_use]
pub fn residual_add() -> Program {
    let mut builder = Builder::new(ProgramKind::Map);
    let residual = builder.input(0);
    let update = builder.input(1);
    let sum = builder.binary(BinOp::Add, residual, update);
    builder.output(0, sum);
    builder.finish()
}

/// Builds Qwen3 half-split rotary position embedding for head dimension 128.
#[must_use]
pub fn half_split_rope() -> Program {
    let mut builder = Builder::new(ProgramKind::Map);
    let first = builder.input(0);
    let second = builder.input(1);
    let position = builder.index(0);
    let position = builder.cast(ValueType::F32, position);
    let frequency = builder.index(3);
    let frequency = builder.cast(ValueType::F32, frequency);
    let two = builder.constant(2.0);
    let doubled = builder.binary(BinOp::Mul, frequency, two);
    let head_dimension = builder.constant(128.0);
    let exponent = builder.binary(BinOp::Div, doubled, head_dimension);
    let theta = builder.constant(1_000_000.0);
    let scale = builder.binary(BinOp::Pow, theta, exponent);
    let angle = builder.binary(BinOp::Div, position, scale);
    let cosine = builder.unary(UnOp::Cos, angle);
    let sine = builder.unary(UnOp::Sin, angle);
    let first_cosine = builder.binary(BinOp::Mul, first, cosine);
    let second_sine = builder.binary(BinOp::Mul, second, sine);
    let first_rotated = builder.binary(BinOp::Sub, first_cosine, second_sine);
    let second_cosine = builder.binary(BinOp::Mul, second, cosine);
    let first_sine = builder.binary(BinOp::Mul, first, sine);
    let second_rotated = builder.binary(BinOp::Add, second_cosine, first_sine);
    builder.output(0, first_rotated);
    builder.output(1, second_rotated);
    builder.finish()
}

struct Builder {
    kind: ProgramKind,
    insts: Vec<Inst>,
    outputs: Vec<(u32, u32)>,
}

impl Builder {
    fn new(kind: ProgramKind) -> Self {
        Self {
            kind,
            insts: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn push(&mut self, inst: Inst) -> u32 {
        let value = u32::try_from(self.insts.len()).unwrap_or(u32::MAX);
        self.insts.push(inst);
        value
    }

    fn input(&mut self, slot: u32) -> u32 {
        self.push(Inst::Input(slot))
    }

    fn constant(&mut self, value: f32) -> u32 {
        self.push(Inst::Const(value))
    }

    fn index(&mut self, axis: u8) -> u32 {
        self.push(Inst::Index(axis))
    }

    fn extent(&mut self, axis: u8) -> u32 {
        self.push(Inst::Extent(axis))
    }

    fn unary(&mut self, op: UnOp, value: u32) -> u32 {
        self.push(Inst::Unary(op, value))
    }

    fn binary(&mut self, op: BinOp, left: u32, right: u32) -> u32 {
        self.push(Inst::Binary(op, left, right))
    }

    fn cast(&mut self, to: ValueType, value: u32) -> u32 {
        self.push(Inst::Cast(to, value))
    }

    fn reduce(&mut self, op: RedOp, value: u32) -> u32 {
        self.push(Inst::Reduce(op, value))
    }

    fn output(&mut self, slot: u32, value: u32) {
        self.outputs.push((slot, value));
    }

    fn finish(self) -> Program {
        Program {
            kind: self.kind,
            insts: self.insts,
            outputs: self.outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use forja_core::{Backend, CommandList, DType, Op, Slice, Submission, Tensor, ViewOp};
    use forja_cpu::CpuBackend;

    use super::*;
    use crate::{DeterministicValues, assert_outputs_agree};

    #[test]
    fn rms_norm_matches_the_trusted_operation() {
        let backend = CpuBackend::new();
        let hidden = initialized(&backend, &[7, 1024], 1);
        let weight = initialized(&backend, &[1024], 2);
        let broadcast_weight = backend
            .view(&weight, ViewOp::Broadcast(vec![7, 1024]))
            .unwrap();
        let expected = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        let actual = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        assert_program_matches(
            &backend,
            Op::RmsNorm { eps: 1e-6 },
            &[&hidden, &weight],
            &expected,
            &rms_norm(1e-6),
            &[&hidden, &broadcast_weight],
            &[&actual],
            &[(&expected, &actual)],
        );
    }

    #[test]
    fn softmax_matches_the_trusted_operation() {
        let backend = CpuBackend::new();
        let scores = initialized(&backend, &[16, 7, 33], 3);
        let expected = backend.alloc(DType::F32, &[16, 7, 33]).unwrap();
        let actual = backend.alloc(DType::F32, &[16, 7, 33]).unwrap();
        assert_program_matches(
            &backend,
            Op::Softmax,
            &[&scores],
            &expected,
            &softmax(),
            &[&scores],
            &[&actual],
            &[(&expected, &actual)],
        );
    }

    #[test]
    fn silu_mul_matches_the_trusted_operation() {
        let backend = CpuBackend::new();
        let gate = initialized(&backend, &[7, 3072], 4);
        let up = initialized(&backend, &[7, 3072], 5);
        let expected = backend.alloc(DType::F32, &[7, 3072]).unwrap();
        let actual = backend.alloc(DType::F32, &[7, 3072]).unwrap();
        assert_program_matches(
            &backend,
            Op::SiluMul,
            &[&gate, &up],
            &expected,
            &silu_mul(),
            &[&gate, &up],
            &[&actual],
            &[(&expected, &actual)],
        );
    }

    #[test]
    fn residual_add_matches_the_trusted_operation() {
        let backend = CpuBackend::new();
        let residual = initialized(&backend, &[7, 1024], 6);
        let update = initialized(&backend, &[7, 1024], 7);
        let expected = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        let actual = backend.alloc(DType::F32, &[7, 1024]).unwrap();
        assert_program_matches(
            &backend,
            Op::Add,
            &[&residual, &update],
            &expected,
            &residual_add(),
            &[&residual, &update],
            &[&actual],
            &[(&expected, &actual)],
        );
    }

    #[test]
    fn half_split_rope_matches_the_trusted_operation() {
        let backend = CpuBackend::new();
        let values = initialized(&backend, &[7, 16, 128], 8);
        let positions = backend.alloc(DType::U32, &[7]).unwrap();
        backend
            .write(
                &positions,
                &(0_u32..7).flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
            )
            .unwrap();
        let reshaped = backend
            .view(&values, ViewOp::Reshape(vec![7, 16, 2, 64]))
            .unwrap();
        let first = half(&backend, &reshaped, 0);
        let second = half(&backend, &reshaped, 1);
        let expected = backend.alloc(DType::F32, &[7, 16, 128]).unwrap();
        let expected_reshaped = backend
            .view(&expected, ViewOp::Reshape(vec![7, 16, 2, 64]))
            .unwrap();
        let expected_first = half(&backend, &expected_reshaped, 0);
        let expected_second = half(&backend, &expected_reshaped, 1);
        let actual_first = backend.alloc(DType::F32, &[7, 16, 1, 64]).unwrap();
        let actual_second = backend.alloc(DType::F32, &[7, 16, 1, 64]).unwrap();
        assert_program_matches(
            &backend,
            Op::Rope { theta: 1_000_000.0 },
            &[&values, &positions],
            &expected,
            &half_split_rope(),
            &[&first, &second],
            &[&actual_first, &actual_second],
            &[
                (&expected_first, &actual_first),
                (&expected_second, &actual_second),
            ],
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_program_matches(
        backend: &CpuBackend,
        trusted: Op,
        trusted_inputs: &[&Tensor],
        trusted_output: &Tensor,
        program: &Program,
        program_inputs: &[&Tensor],
        program_outputs: &[&Tensor],
        comparisons: &[(&Tensor, &Tensor)],
    ) {
        let mut trusted_commands = CommandList::new();
        trusted_commands
            .dispatch(trusted, trusted_inputs, trusted_output)
            .unwrap();
        backend.submit(trusted_commands).unwrap().wait().unwrap();

        let program = program.validate().unwrap();
        let mut program_commands = CommandList::new();
        program_commands
            .dispatch_program(&program, program_inputs, program_outputs)
            .unwrap();
        backend.submit(program_commands).unwrap().wait().unwrap();

        for &(expected, actual) in comparisons {
            assert_outputs_agree(
                expected.layout().dtype(),
                &backend.read(expected).unwrap(),
                &backend.read(actual).unwrap(),
            )
            .unwrap();
        }
    }

    fn initialized(backend: &CpuBackend, shape: &[u32], seed: u64) -> Tensor {
        let tensor = backend.alloc(DType::F32, shape).unwrap();
        let mut values = DeterministicValues::new(seed);
        let bytes = (0..tensor.layout().element_count())
            .flat_map(|_| values.next_f32().to_le_bytes())
            .collect::<Vec<_>>();
        backend.write(&tensor, &bytes).unwrap();
        tensor
    }

    fn half(backend: &CpuBackend, tensor: &Tensor, index: u32) -> Tensor {
        backend
            .view(
                tensor,
                ViewOp::Slice(vec![
                    Slice::new(0, 7, 1).unwrap(),
                    Slice::new(0, 16, 1).unwrap(),
                    Slice::new(index, 1, 1).unwrap(),
                    Slice::new(0, 64, 1).unwrap(),
                ]),
            )
            .unwrap()
    }
}
