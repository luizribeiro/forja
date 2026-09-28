//! Foundations used by Rust-syntax kernel definitions.
//!
//! Generated allocating calls broadcast non-leading inputs to the leading
//! input's concrete shape. Graph replay must refuse that automatic broadcast
//! when an extent is symbolic; callers can use the generated `_into` function
//! with explicitly bound views instead.
//!
//! Kernel Boolean operators and conditionals evaluate both sides before a
//! select, so they do not short-circuit.

use std::{cell::RefCell, collections::VecDeque, marker::PhantomData};

use crate::{DType, Element, Error, FloatElement, Result, Tensor, graph, sys};

pub use crate::program::Kernel;
pub use forja_sdk_macros::kernel;

/// Maximum number of prepared signatures retained by one kernel definition.
pub const CACHE_CAPACITY: usize = 16;

/// A type-erased tensor binding used by generated kernels.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct TensorRef<'a> {
    handle: &'a sys::Handle,
    shape: &'a [u32],
    dtype: DType,
}

impl<'a> TensorRef<'a> {
    /// Erases a tensor's storage element while retaining its checked metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor's element type has no kernel dtype.
    pub fn new<T: Element>(tensor: &'a Tensor<T>) -> Result<Self> {
        Ok(Self {
            handle: tensor.handle(),
            shape: tensor.shape(),
            dtype: sys::dtype(T::DTYPE)?,
        })
    }

    /// Returns the tensor's storage type.
    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }
}

/// Dispatches a generated kernel into newly allocated outputs.
#[doc(hidden)]
pub fn run<T: FloatElement, const N: usize>(
    kernel: &Kernel,
    inputs: &[TensorRef<'_>],
) -> Result<[Tensor<T>; N]> {
    let shape = inputs
        .first()
        .ok_or_else(|| Error::new("kernel requires at least one tensor input"))?
        .shape;
    let outputs = (0..N)
        .map(|_| Tensor::<T>::empty(shape.to_vec()))
        .collect::<Result<Vec<_>>>()?;
    let output_refs = outputs
        .iter()
        .map(TensorRef::new)
        .collect::<Result<Vec<_>>>()?;
    run_into(kernel, inputs, &output_refs)?;
    outputs
        .try_into()
        .map_err(|_| Error::new("kernel allocated an unexpected output count"))
}

/// Dispatches a generated kernel into caller-supplied output views.
#[doc(hidden)]
pub fn run_into(
    kernel: &Kernel,
    inputs: &[TensorRef<'_>],
    outputs: &[TensorRef<'_>],
) -> Result<()> {
    check_bindings(kernel, inputs, outputs)?;
    let input_handles = inputs
        .iter()
        .map(|binding| binding.handle)
        .collect::<Vec<_>>();
    let output_handles = outputs
        .iter()
        .map(|binding| binding.handle)
        .collect::<Vec<_>>();
    graph::record_kernel(&kernel.handle, &input_handles, &output_handles)
}

fn check_bindings(
    kernel: &Kernel,
    inputs: &[TensorRef<'_>],
    outputs: &[TensorRef<'_>],
) -> Result<()> {
    let rank = inputs
        .first()
        .and_then(|input| u8::try_from(input.shape.len()).ok())
        .ok_or_else(|| Error::new("kernel tensor rank is invalid"))?;
    let input_dtypes = inputs.iter().map(|input| input.dtype).collect::<Vec<_>>();
    let output_dtypes = outputs
        .iter()
        .map(|output| output.dtype)
        .collect::<Vec<_>>();
    if rank != kernel.rank || input_dtypes != kernel.inputs || output_dtypes != kernel.outputs {
        return Err(Error::new(
            "kernel tensor bindings do not match its signature",
        ));
    }
    Ok(())
}

/// A row-kernel tensor parameter in `#[kernel]` syntax.
///
/// `T` is the lane's compute type and defaults to `f32`; storage precision is
/// selected when the generated kernel is called. This zero-sized marker is
/// syntax consumed by the kernel attribute and does not contain tensor data.
#[derive(Clone, Copy, Debug)]
pub struct Row<T = f32>(PhantomData<fn() -> T>);

/// An elementwise map-kernel tensor parameter in `#[kernel]` syntax.
///
/// `T` is the element's compute type and defaults to `f32`; storage precision
/// is selected when the generated kernel is called. This zero-sized marker is
/// syntax consumed by the kernel attribute and does not contain tensor data.
#[derive(Clone, Copy, Debug)]
pub struct Elem<T = f32>(PhantomData<fn() -> T>);

/// Names an iteration-axis coordinate in `#[kernel]` syntax.
///
/// The kernel attribute requires a literal axis and lowers this call to an
/// unsigned IR index; negative axes count from the end. Outside an attributed
/// kernel this syntax-only placeholder has no coordinate context.
#[must_use]
pub const fn index(_axis: i32) -> u32 {
    0
}

/// Names an iteration-axis extent in `#[kernel]` syntax.
///
/// The kernel attribute requires a literal axis and lowers this call to an
/// unsigned IR extent; negative axes count from the end. Outside an attributed
/// kernel this syntax-only placeholder has no shape context.
#[must_use]
pub const fn extent(_axis: i32) -> u32 {
    0
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Signature {
    rank: u8,
    inputs: Vec<DType>,
    outputs: Vec<DType>,
    scalars: Vec<u32>,
}

struct Entry {
    signature: Signature,
    kernel: Kernel,
}

/// A bounded cache of prepared handles for one kernel definition.
///
/// Signatures include tensor rank, input and output storage types, and the bit
/// patterns of scalar constants compiled into the program. Least-recently used
/// entries are evicted, dropping the cache's clone of the prepared handle.
pub struct Cache {
    entries: RefCell<VecDeque<Entry>>,
    full: FullBehavior,
}

#[derive(Clone, Copy)]
enum FullBehavior {
    Evict,
    Reject(&'static str),
}

impl Cache {
    /// Creates an empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
            full: FullBehavior::Evict,
        }
    }

    /// Creates an empty cache that rejects signatures beyond its capacity.
    #[doc(hidden)]
    #[must_use]
    pub const fn rejecting(kernel_name: &'static str) -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
            full: FullBehavior::Reject(kernel_name),
        }
    }

    /// Returns the cached handle or builds and retains one on a miss.
    ///
    /// The builder is called at most once and only after confirming a miss. A
    /// failed build leaves existing entries untouched.
    ///
    /// # Errors
    ///
    /// Returns the error produced while preparing a missing signature.
    pub fn get_or_try_insert_with(
        &self,
        rank: u8,
        input_dtypes: &[DType],
        output_dtypes: &[DType],
        scalar_bits: &[u32],
        build: impl FnOnce() -> Result<Kernel>,
    ) -> Result<Kernel> {
        let signature = Signature {
            rank,
            inputs: input_dtypes.to_vec(),
            outputs: output_dtypes.to_vec(),
            scalars: scalar_bits.to_vec(),
        };
        let mut entries = self.entries.borrow_mut();
        if let Some(index) = entries
            .iter()
            .position(|entry| entry.signature == signature)
        {
            let entry = entries
                .remove(index)
                .ok_or_else(|| crate::Error::new("kernel cache entry disappeared"))?;
            let kernel = entry.kernel.clone();
            entries.push_back(entry);
            return Ok(kernel);
        }
        drop(entries);

        let mut entries = self.entries.borrow_mut();
        if entries.len() == CACHE_CAPACITY {
            match self.full {
                FullBehavior::Evict => {
                    entries.pop_front();
                }
                FullBehavior::Reject(name) => {
                    return Err(crate::Error::new(format!(
                        "kernel `{name}`: more than 16 distinct (rank, dtype, scalar) variants; pass varying scalars as tensors"
                    )));
                }
            }
        }
        drop(entries);

        let kernel = build()?;
        let mut entries = self.entries.borrow_mut();
        entries.push_back(Entry {
            signature,
            kernel: kernel.clone(),
        });
        Ok(kernel)
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use super::*;
    use crate::program::Ctx;

    fn kernel(rank: u8, inputs: &[DType], outputs: &[DType]) -> Result<Kernel> {
        let context = Ctx::new();
        context.output(0, context.input(0));
        Kernel::new(&context, rank, inputs, outputs)
    }

    #[test]
    fn caches_hits_and_misses_by_full_signature() {
        let cache = Cache::new();
        let builds = Cell::new(0);
        let get = |rank, inputs: &[DType], outputs: &[DType], scalars: &[u32]| {
            cache.get_or_try_insert_with(rank, inputs, outputs, scalars, || {
                builds.set(builds.get() + 1);
                kernel(rank, inputs, outputs)
            })
        };

        let first = get(1, &[DType::F32], &[DType::F32], &[1]).unwrap();
        let hit = get(1, &[DType::F32], &[DType::F32], &[1]).unwrap();
        let scalar_miss = get(1, &[DType::F32], &[DType::F32], &[2]).unwrap();
        get(2, &[DType::F32], &[DType::F32], &[1]).unwrap();
        get(1, &[DType::BF16], &[DType::F32], &[1]).unwrap();
        get(1, &[DType::F32], &[DType::BF16], &[1]).unwrap();

        assert!(Rc::ptr_eq(&first.handle, &hit.handle));
        assert!(!Rc::ptr_eq(&first.handle, &scalar_miss.handle));
        assert_eq!(builds.get(), 5);
    }

    #[test]
    fn distinguishes_signed_zero_and_nan_payloads() {
        let cache = Cache::new();
        let builds = Cell::new(0);
        let values = [
            -0.0_f32,
            0.0_f32,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0x7fc0_0002),
        ];

        for value in values.into_iter().chain(values) {
            cache
                .get_or_try_insert_with(1, &[DType::F32], &[DType::F32], &[value.to_bits()], || {
                    builds.set(builds.get() + 1);
                    kernel(1, &[DType::F32], &[DType::F32])
                })
                .unwrap();
        }

        assert_eq!(builds.get(), values.len());
    }

    #[test]
    fn eviction_releases_the_cached_handle() {
        let cache = Cache::new();
        let first = cache
            .get_or_try_insert_with(1, &[DType::F32], &[DType::F32], &[0], || {
                kernel(1, &[DType::F32], &[DType::F32])
            })
            .unwrap();
        let handle = Rc::downgrade(&first.handle);
        drop(first);

        for scalar in 1..=CACHE_CAPACITY {
            cache
                .get_or_try_insert_with(
                    1,
                    &[DType::F32],
                    &[DType::F32],
                    &[u32::try_from(scalar).unwrap()],
                    || kernel(1, &[DType::F32], &[DType::F32]),
                )
                .unwrap();
        }

        assert!(handle.upgrade().is_none());
    }

    #[test]
    fn rejecting_cache_reports_the_seventeenth_signature() {
        let cache = Cache::rejecting("position_kernel");
        for scalar in 0..CACHE_CAPACITY {
            cache
                .get_or_try_insert_with(
                    1,
                    &[DType::F32],
                    &[DType::F32],
                    &[u32::try_from(scalar).unwrap()],
                    || kernel(1, &[DType::F32], &[DType::F32]),
                )
                .unwrap();
        }

        let error = cache
            .get_or_try_insert_with(
                1,
                &[DType::F32],
                &[DType::F32],
                &[u32::try_from(CACHE_CAPACITY).unwrap()],
                || kernel(1, &[DType::F32], &[DType::F32]),
            )
            .err()
            .expect("the seventeenth signature must be rejected");
        assert_eq!(
            error.to_string(),
            "kernel `position_kernel`: more than 16 distinct (rank, dtype, scalar) variants; pass varying scalars as tensors"
        );
    }

    #[test]
    fn dispatches_mixed_storage_types() {
        let context = Ctx::new();
        context.output(0, context.input(0) + context.input_u32(1).cast_f32());
        let kernel = Kernel::new(&context, 1, &[DType::F32, DType::U32], &[DType::F32]).unwrap();
        let floats = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
        let integers = Tensor::from_slice(&[3_u32, 4], &[2]).unwrap();
        let inputs = [
            TensorRef::new(&floats).unwrap(),
            TensorRef::new(&integers).unwrap(),
        ];
        let [output] = run::<f32, 1>(&kernel, &inputs).unwrap();

        assert_eq!(output.to_vec().unwrap(), [4.0, 6.0]);
    }
}
