//! Foundations used by Rust-syntax kernel definitions.

use std::{cell::RefCell, collections::VecDeque, marker::PhantomData};

use crate::{DType, Result};

pub use crate::program::Kernel;

/// Maximum number of prepared signatures retained by one kernel definition.
pub const CACHE_CAPACITY: usize = 16;

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
}

impl Cache {
    /// Creates an empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: RefCell::new(VecDeque::new()),
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

        let kernel = build()?;
        let mut entries = self.entries.borrow_mut();
        if entries.len() == CACHE_CAPACITY {
            entries.pop_front();
        }
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
}
