//! Reusable transformer blocks.

use crate::{Dim, Element, Error, Result, Tensor};

/// Fixed-capacity key/value cache with `[heads, context, width]` storage.
pub struct KvCache<T: Element> {
    key: Tensor<T>,
    value: Tensor<T>,
}

impl<T: Element> KvCache<T> {
    /// Allocates a zero-filled cache.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache size overflows or allocation is refused.
    pub fn new(heads: u32, context: u32, width: u32, zero: T) -> Result<Self> {
        let count = u64::from(heads)
            .checked_mul(u64::from(context))
            .and_then(|count| count.checked_mul(u64::from(width)))
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| Error::loading("KV cache size does not fit usize"))?;
        let zeros = vec![zero; count];
        let shape = [heads, context, width];
        Ok(Self {
            key: Tensor::from_slice(&zeros, &shape)?,
            value: Tensor::from_slice(&zeros, &shape)?,
        })
    }

    /// Appends projected keys and values and returns cache prefixes ending at `end`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid ranges, shapes, aliasing, or refused copies.
    pub fn append(
        &mut self,
        key: &Tensor<T>,
        value: &Tensor<T>,
        start: &Dim,
        sequence: u32,
        end: &Dim,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        key.copy_into(&mut self.key.narrow(1, start, sequence)?)?;
        value.copy_into(&mut self.value.narrow(1, start, sequence)?)?;
        Ok((self.key.narrow(1, 0, end)?, self.value.narrow(1, 0, end)?))
    }
}
