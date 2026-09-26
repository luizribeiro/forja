use std::{error::Error, fmt};

use crate::{DType, Layout, byte_ranges_overlap, is_injective};

/// An opaque allocation identity unique to one backend instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferId {
    backend: u64,
    allocation: u64,
    byte_len: u64,
}

impl BufferId {
    /// Creates an allocation identity for use by a backend implementation.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(backend: u64, allocation: u64, byte_len: u64) -> Self {
        Self {
            backend,
            allocation,
            byte_len,
        }
    }

    /// Returns the backend instance identity.
    #[doc(hidden)]
    #[must_use]
    pub const fn backend(self) -> u64 {
        self.backend
    }
}

/// A reason tensor construction failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorError {
    /// The layout was validated against a different buffer length.
    BufferLengthMismatch,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("layout and buffer lengths differ")
    }
}

impl Error for TensorError {}

/// A validated tensor view into a backend allocation.
///
/// Backends re-check the allocation identity and length whenever they receive
/// a tensor. Guests only reach tensors through resource handles maintained by
/// the host, rather than constructing this type directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Tensor {
    buffer: BufferId,
    layout: Layout,
}

impl Tensor {
    /// Creates a tensor for use by a backend implementation.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if the layout was validated for another length.
    #[doc(hidden)]
    pub fn new(buffer: BufferId, layout: Layout) -> Result<Self, TensorError> {
        if buffer.byte_len != layout.buffer_len() {
            return Err(TensorError::BufferLengthMismatch);
        }
        Ok(Self { buffer, layout })
    }

    /// Returns the allocation identity.
    #[must_use]
    pub const fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// Returns the validated layout.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// A trusted operation recorded in a command list.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Copies and optionally casts one tensor.
    Copy,
    /// Adds two tensors elementwise.
    Add,
    /// Applies `SiLU` to a gate and multiplies it by an up tensor.
    SiluMul,
    /// Normalizes rows by their root mean square and applies a weight.
    RmsNorm {
        /// The nonnegative stabilizer added before the square root.
        eps: f32,
    },
    /// Applies stable softmax over the last axis.
    ///
    /// A row containing only negative infinity produces all zeros.
    Softmax,
}

/// An operand named by an operation validation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    /// The input at this zero-based position.
    Input(usize),
    /// The output tensor.
    Output,
}

/// A reason an operation could not be recorded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpError {
    /// The operation received the wrong number of inputs.
    Arity {
        /// The required count.
        expected: usize,
        /// The received count.
        actual: usize,
    },
    /// An operand has a scalar type unsupported by the operation.
    DType {
        /// The offending operand.
        operand: Operand,
        /// Its scalar type.
        dtype: DType,
    },
    /// An operand shape differs from the required shape.
    Shape {
        /// The offending operand.
        operand: Operand,
    },
    /// The output maps multiple logical elements to the same storage.
    NonInjectiveOutput,
    /// An input and output may touch the same bytes.
    Aliasing {
        /// The overlapping input position.
        input: usize,
    },
    /// An RMS normalization epsilon is negative or non-finite.
    InvalidEpsilon,
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid operation: {self:?}")
    }
}

impl Error for OpError {}

/// A dispatch whose signature and aliasing have been validated.
#[derive(Clone, Debug)]
pub struct Dispatch {
    op: Op,
    inputs: Vec<Tensor>,
    output: Tensor,
}

impl Dispatch {
    /// Returns the operation configuration.
    #[must_use]
    pub const fn op(&self) -> Op {
        self.op
    }
    /// Returns the input tensors.
    #[must_use]
    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
    /// Returns the output tensor.
    #[must_use]
    pub const fn output(&self) -> &Tensor {
        &self.output
    }
}

/// An ordered list of validated backend work.
#[derive(Clone, Debug, Default)]
pub struct CommandList {
    dispatches: Vec<Dispatch>,
}

impl CommandList {
    /// Creates an empty command list.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dispatches: Vec::new(),
        }
    }

    /// Validates and records a dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`OpError`] for an invalid signature or unsafe aliasing.
    pub fn dispatch(&mut self, op: Op, inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
        check_common(inputs, output)?;
        match op {
            Op::Copy => check_copy(inputs, output)?,
            Op::Add | Op::SiluMul => check_binary(inputs, output)?,
            Op::RmsNorm { eps } => check_rms_norm(inputs, output, eps)?,
            Op::Softmax => check_softmax(inputs, output)?,
        }
        self.dispatches.push(Dispatch {
            op,
            inputs: inputs.iter().map(|tensor| (*tensor).clone()).collect(),
            output: output.clone(),
        });
        Ok(())
    }

    /// Consumes the list into its validated dispatches.
    #[must_use]
    pub fn into_dispatches(self) -> Vec<Dispatch> {
        self.dispatches
    }
}

fn check_common(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if !is_injective(output.layout()) {
        return Err(OpError::NonInjectiveOutput);
    }
    for (input, tensor) in inputs.iter().enumerate() {
        if tensor.buffer == output.buffer && byte_ranges_overlap(tensor.layout(), output.layout()) {
            return Err(OpError::Aliasing { input });
        }
    }
    Ok(())
}

fn check_copy(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(output, Operand::Output)?;
    check_shape(output, inputs[0], Operand::Output)?;
    Ok(())
}

fn check_binary(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    check_shape(inputs[1], inputs[0], Operand::Input(1))?;
    check_shape(output, inputs[0], Operand::Output)
}

fn check_shape(tensor: &Tensor, expected: &Tensor, operand: Operand) -> Result<(), OpError> {
    if tensor.layout.shape() != expected.layout.shape() {
        return Err(OpError::Shape { operand });
    }
    Ok(())
}

fn check_rms_norm(inputs: &[&Tensor], output: &Tensor, eps: f32) -> Result<(), OpError> {
    if inputs.len() != 2 {
        return Err(OpError::Arity {
            expected: 2,
            actual: inputs.len(),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OpError::InvalidEpsilon);
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(inputs[1], Operand::Input(1))?;
    check_float(output, Operand::Output)?;
    let Some(&width) = inputs[0].layout.shape().last() else {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    };
    if width == 0 || inputs[1].layout.shape() != [width] {
        return Err(OpError::Shape {
            operand: Operand::Input(1),
        });
    }
    check_shape(output, inputs[0], Operand::Output)
}

fn check_softmax(inputs: &[&Tensor], output: &Tensor) -> Result<(), OpError> {
    if inputs.len() != 1 {
        return Err(OpError::Arity {
            expected: 1,
            actual: inputs.len(),
        });
    }
    check_float(inputs[0], Operand::Input(0))?;
    check_float(output, Operand::Output)?;
    if inputs[0]
        .layout
        .shape()
        .last()
        .is_none_or(|&width| width == 0)
    {
        return Err(OpError::Shape {
            operand: Operand::Input(0),
        });
    }
    check_shape(output, inputs[0], Operand::Output)
}

fn check_float(tensor: &Tensor, operand: Operand) -> Result<(), OpError> {
    match tensor.layout.dtype() {
        DType::F32 | DType::F16 | DType::BF16 => Ok(()),
        dtype => Err(OpError::DType { operand, dtype }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(buffer: u64, dtype: DType, shape: &[u32], strides: &[u64]) -> Tensor {
        let bytes = u64::from(shape.iter().product::<u32>()) * dtype.byte_size();
        Tensor::new(
            BufferId::new(1, buffer, bytes),
            Layout::new(dtype, 0, shape.to_vec(), strides.to_vec(), bytes).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn copy_rejects_each_invalid_signature() {
        let input = tensor(1, DType::F32, &[2], &[1]);
        let output = tensor(2, DType::F32, &[2], &[0]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &output),
            Err(OpError::NonInjectiveOutput)
        );
        let overlap = Tensor::new(
            input.buffer(),
            Layout::contiguous(DType::F32, 0, vec![2], 8).unwrap(),
        )
        .unwrap();
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &overlap),
            Err(OpError::Aliasing { input: 0 })
        );
        let integer = tensor(3, DType::I32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&integer], &input),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::I32
            })
        );
        let wrong_shape = tensor(4, DType::F32, &[1], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&input], &wrong_shape),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn add_rejects_invalid_signatures() {
        let input = tensor(1, DType::F32, &[2], &[1]);
        let integer = tensor(2, DType::U32, &[2], &[1]);
        let wrong_shape = tensor(3, DType::F32, &[1], &[1]);
        let output = tensor(4, DType::F32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input, &integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Add, &[&input, &wrong_shape], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
    }

    #[test]
    fn silu_mul_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[2], &[1]);
        let integer = tensor(2, DType::I32, &[2], &[1]);
        let output = tensor(3, DType::BF16, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(Op::SiluMul, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::SiluMul, &[&input, &integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(1),
                dtype: DType::I32
            })
        );
    }

    #[test]
    fn rms_norm_rejects_invalid_signatures() {
        let input = tensor(1, DType::F32, &[3, 2], &[2, 1]);
        let weight = tensor(2, DType::F32, &[3], &[1]);
        let output = tensor(3, DType::F32, &[3, 2], &[2, 1]);
        let integer = tensor(4, DType::I32, &[3, 2], &[2, 1]);
        let wrong_output = tensor(5, DType::F32, &[3, 1], &[1, 1]);
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&input], &output),
            Err(OpError::Arity {
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: -1.0 }, &[&input, &weight], &output),
            Err(OpError::InvalidEpsilon)
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&input, &weight], &output),
            Err(OpError::Shape {
                operand: Operand::Input(1)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::RmsNorm { eps: 0.0 }, &[&integer, &weight], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::I32
            })
        );
        let good_weight = tensor(6, DType::F32, &[2], &[1]);
        assert_eq!(
            CommandList::new().dispatch(
                Op::RmsNorm { eps: 0.0 },
                &[&input, &good_weight],
                &wrong_output
            ),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }

    #[test]
    fn softmax_rejects_invalid_signatures() {
        let input = tensor(1, DType::F16, &[2], &[1]);
        let integer = tensor(2, DType::U32, &[2], &[1]);
        let output = tensor(3, DType::F32, &[2], &[1]);
        let scalar = tensor(4, DType::F32, &[], &[]);
        let scalar_output = tensor(5, DType::F32, &[], &[]);
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[], &output),
            Err(OpError::Arity {
                expected: 1,
                actual: 0
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&integer], &output),
            Err(OpError::DType {
                operand: Operand::Input(0),
                dtype: DType::U32
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&scalar], &scalar_output),
            Err(OpError::Shape {
                operand: Operand::Input(0)
            })
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Softmax, &[&input], &scalar),
            Err(OpError::Shape {
                operand: Operand::Output
            })
        );
    }
}
