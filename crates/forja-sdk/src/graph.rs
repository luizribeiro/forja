use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use crate::{Result, sys};

thread_local! {
    static CURRENT: RefCell<Option<sys::Commands>> = const { RefCell::new(None) };
    static KERNELS: RefCell<VecDeque<CachedKernel>> = const { RefCell::new(VecDeque::new()) };
}

const KERNEL_CACHE_CAPACITY: usize = 8;

#[derive(Eq, PartialEq)]
struct KernelKey {
    source: Vec<u8>,
    rank: u8,
    inputs: Vec<sys::DType>,
    outputs: Vec<sys::DType>,
}

struct CachedKernel {
    key: KernelKey,
    kernel: Rc<sys::Kernel>,
}

pub(crate) fn record(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    output: &sys::Handle,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(sys::command_list()?);
        }
        let commands = current
            .as_mut()
            .ok_or_else(|| crate::Error::new("current graph was not initialized"))?;
        sys::dispatch(commands, operation, inputs, output)
    })
}

pub(crate) fn record_program(
    program: sys::Program,
    rank: u8,
    input_dtypes: Vec<sys::DType>,
    output_dtypes: Vec<sys::DType>,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    let kernel = cached_kernel(program, rank, input_dtypes, output_dtypes)?;
    record_kernel(&kernel, inputs, outputs)
}

pub(crate) fn record_kernel(
    kernel: &sys::Kernel,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(sys::command_list()?);
        }
        let commands = current
            .as_mut()
            .ok_or_else(|| crate::Error::new("current graph was not initialized"))?;
        sys::dispatch_kernel(commands, kernel, inputs, outputs)
    })
}

fn cached_kernel(
    program: sys::Program,
    rank: u8,
    inputs: Vec<sys::DType>,
    outputs: Vec<sys::DType>,
) -> Result<Rc<sys::Kernel>> {
    let key = KernelKey {
        source: program_bytes(&program),
        rank,
        inputs,
        outputs,
    };
    KERNELS.with(|kernels| {
        let mut kernels = kernels.borrow_mut();
        if let Some(index) = kernels.iter().position(|entry| entry.key == key)
            && let Some(entry) = kernels.remove(index)
        {
            let kernel = Rc::clone(&entry.kernel);
            kernels.push_back(entry);
            return Ok(kernel);
        }
        if kernels.len() == KERNEL_CACHE_CAPACITY {
            kernels.pop_front();
        }
        let kernel = Rc::new(sys::create_kernel(
            program,
            rank,
            &key.inputs,
            &key.outputs,
        )?);
        kernels.push_back(CachedKernel {
            key,
            kernel: Rc::clone(&kernel),
        });
        Ok(kernel)
    })
}

fn program_bytes(program: &sys::Program) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(match program.kind {
        crate::program::ProgramKind::Map => 0,
        crate::program::ProgramKind::Row => 1,
    });
    bytes.extend(program.instructions.len().to_le_bytes());
    for instruction in &program.instructions {
        encode_instruction(&mut bytes, *instruction);
    }
    bytes.push(0xff);
    for &(slot, value) in &program.outputs {
        bytes.extend(slot.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

#[allow(clippy::too_many_lines)]
fn encode_instruction(bytes: &mut Vec<u8>, instruction: sys::ProgramInst) {
    use sys::ProgramInst;
    match instruction {
        ProgramInst::Input(slot) => encode_one(bytes, 0, slot),
        ProgramInst::Constant(value) => encode_one(bytes, 1, value.to_bits()),
        ProgramInst::Index(axis) => encode_axis(bytes, 2, axis),
        ProgramInst::Extent(axis) => encode_axis(bytes, 3, axis),
        ProgramInst::Unary(op, value) => {
            bytes.extend([4, op as u8]);
            bytes.extend(value.to_le_bytes());
        }
        ProgramInst::Binary(op, left, right) => {
            bytes.extend([5, op as u8]);
            bytes.extend(left.to_le_bytes());
            bytes.extend(right.to_le_bytes());
        }
        ProgramInst::Select(condition, accepted, rejected) => {
            bytes.push(6);
            bytes.extend(condition.to_le_bytes());
            bytes.extend(accepted.to_le_bytes());
            bytes.extend(rejected.to_le_bytes());
        }
        ProgramInst::CastF32(value) => encode_one(bytes, 7, value),
        ProgramInst::Reduce(op, value) => {
            bytes.extend([8, op as u8]);
            bytes.extend(value.to_le_bytes());
        }
    }
}

fn encode_one(bytes: &mut Vec<u8>, tag: u8, value: u32) {
    bytes.push(tag);
    bytes.extend(value.to_le_bytes());
}

fn encode_axis(bytes: &mut Vec<u8>, tag: u8, axis: u8) {
    bytes.extend([tag, axis]);
}

#[cfg(feature = "native")]
pub(crate) fn clear_kernel_cache() {
    KERNELS.with(|kernels| kernels.borrow_mut().clear());
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::program::ProgramKind;

    fn program(value: f32) -> sys::Program {
        sys::Program {
            kind: ProgramKind::Map,
            instructions: vec![
                sys::ProgramInst::Input(0),
                sys::ProgramInst::Constant(value),
                sys::ProgramInst::Binary(crate::program::BinaryOp::Mul, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
    }

    #[test]
    fn program_cache_reuses_handles_and_evicts_at_its_bound() {
        clear_kernel_cache();
        let first = cached_kernel(
            program(1.0),
            1,
            vec![sys::DType::F32],
            vec![sys::DType::F32],
        )
        .unwrap();
        let repeated = cached_kernel(
            program(1.0),
            1,
            vec![sys::DType::F32],
            vec![sys::DType::F32],
        )
        .unwrap();
        assert!(Rc::ptr_eq(&first, &repeated));

        for value in 2_u16..=9 {
            cached_kernel(
                program(f32::from(value)),
                1,
                vec![sys::DType::F32],
                vec![sys::DType::F32],
            )
            .unwrap();
        }
        KERNELS.with(|kernels| assert_eq!(kernels.borrow().len(), KERNEL_CACHE_CAPACITY));
    }
}

/// Submits all operations recorded by the current thread.
///
/// # Errors
///
/// Returns a host validation or execution error.
pub fn eval() -> Result<()> {
    CURRENT.with(|current| match current.borrow_mut().take() {
        Some(commands) => sys::submit(commands),
        None => Ok(()),
    })
}
