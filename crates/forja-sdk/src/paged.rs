use std::marker::PhantomData;

use crate::{Dim, Element, Error, Result, Tensor};

/// A borrowed capability for the host-owned paged-storage pool.
pub struct PagedPool<'a> {
    #[cfg(target_family = "wasm")]
    raw: &'a crate::sys::guest::paged_storage::PagePool,
    lifetime: PhantomData<&'a ()>,
}

impl PagedPool<'_> {
    #[cfg(target_family = "wasm")]
    #[doc(hidden)]
    #[must_use]
    pub const fn from_guest(raw: &crate::sys::guest::paged_storage::PagePool) -> PagedPool<'_> {
        PagedPool {
            raw,
            lifetime: PhantomData,
        }
    }

    /// Allocates one opaque slot and assigns a key/value plane pair to each layer.
    ///
    /// # Errors
    ///
    /// Returns an error when the layer count overflows or the host refuses the slot.
    pub fn cache_set(self, layers: u32) -> Result<PagedCacheSet> {
        let planes = layers
            .checked_mul(2)
            .ok_or_else(|| Error::loading("paged cache plane count overflowed"))?;
        if planes == 0 {
            return Err(Error::loading("paged cache requires at least one layer"));
        }
        #[cfg(target_family = "wasm")]
        let slot = self
            .raw
            .open_slot(planes)
            .map_err(|error| Error::loading(format!("{error:?}")))?;
        Ok(PagedCacheSet {
            #[cfg(target_family = "wasm")]
            _slot: slot,
            layers,
        })
    }
}

/// Engine-owned composition of one host slot into per-layer key/value caches.
pub struct PagedCacheSet {
    #[cfg(target_family = "wasm")]
    pub(crate) _slot: crate::sys::guest::paged_storage::BatchSlot,
    layers: u32,
}

impl PagedCacheSet {
    /// Returns the number of layer cache pairs.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.layers
    }

    /// Reports whether the set contains no layer cache pairs.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.layers == 0
    }

    /// Borrows one layer's opaque key/value plane pair.
    ///
    /// # Errors
    ///
    /// Returns an error when `layer` is outside this set.
    pub fn layer(&self, layer: u32) -> Result<PagedCache<'_>> {
        if layer >= self.layers {
            return Err(Error::loading("paged cache layer is out of range"));
        }
        let key_plane = layer
            .checked_mul(2)
            .ok_or_else(|| Error::loading("paged cache plane index overflowed"))?;
        Ok(PagedCache {
            _set: self,
            key_plane,
            value_plane: key_plane + 1,
        })
    }
}

/// A borrowed key/value plane pair for one transformer layer.
pub struct PagedCache<'a> {
    _set: &'a PagedCacheSet,
    key_plane: u32,
    value_plane: u32,
}

impl PagedCache<'_> {
    /// Returns the key plane's descriptor index.
    #[must_use]
    pub const fn key_plane(&self) -> u32 {
        self.key_plane
    }

    /// Returns the value plane's descriptor index.
    #[must_use]
    pub const fn value_plane(&self) -> u32 {
        self.value_plane
    }

    /// Appends sequence-major key and value rows to this layer's planes.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible shapes, overflow, or refused host work.
    pub fn append<T: Element>(
        &self,
        key: &Tensor<T>,
        value: &Tensor<T>,
        start: &Dim,
    ) -> Result<()> {
        let [tokens, heads, width]: [u32; 3] = key
            .shape()
            .try_into()
            .map_err(|_| Error::loading("paged key must have rank three"))?;
        if value.shape() != key.shape() {
            return Err(Error::loading("paged key and value shapes differ"));
        }
        let elements = heads
            .checked_mul(width)
            .ok_or_else(|| Error::loading("paged cache row width overflowed"))?;
        let key = key.contiguous()?.reshape(&[tokens, elements])?;
        let value = value.contiguous()?.reshape(&[tokens, elements])?;
        #[cfg(target_family = "wasm")]
        let start = crate::graph::affine(start)?;
        #[cfg(target_family = "wasm")]
        crate::graph::record_import(|commands| {
            crate::sys::paged_copy(
                commands,
                &self._set._slot,
                self.key_plane,
                start,
                u64::from(tokens),
                key.handle(),
            )?;
            crate::sys::paged_copy(
                commands,
                &self._set._slot,
                self.value_plane,
                start,
                u64::from(tokens),
                value.handle(),
            )
        })?;
        #[cfg(not(target_family = "wasm"))]
        let _ = (&key, &value, start);
        Ok(())
    }

    /// Applies causal attention to this layer's host-bound page table.
    ///
    /// The query has shape `[sequence, query_heads, width]`; the result has
    /// shape `[sequence, query_heads * width]`.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible shapes, overflow, or refused host work.
    pub fn attention<T: Element>(
        &self,
        query: &Tensor<T>,
        scale: f32,
        start: &Dim,
        end: &Dim,
    ) -> Result<Tensor<T>> {
        let [sequence, heads, width]: [u32; 3] = query
            .shape()
            .try_into()
            .map_err(|_| Error::loading("paged attention query must have rank three"))?;
        let hidden = heads
            .checked_mul(width)
            .ok_or_else(|| Error::loading("paged attention width overflowed"))?;
        let query = query.permute(&[1, 0, 2])?.contiguous()?;
        let output = Tensor::zeros(&[heads, sequence, width])?;
        #[cfg(target_family = "wasm")]
        let (start, end) = (crate::graph::affine(start)?, crate::graph::affine(end)?);
        #[cfg(target_family = "wasm")]
        crate::graph::record_import(|commands| {
            crate::sys::paged_sdpa(
                commands,
                &self._set._slot,
                query.handle(),
                output.handle(),
                self.key_plane,
                self.value_plane,
                scale,
                true,
                start,
                end,
            )
        })?;
        #[cfg(not(target_family = "wasm"))]
        let _ = (&query, scale, start, end);
        output
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[sequence, hidden])
    }
}

#[cfg(test)]
mod tests {
    use super::PagedCacheSet;

    #[test]
    fn assigns_one_key_value_pair_per_layer() {
        let caches = PagedCacheSet { layers: 3 };
        let middle = caches.layer(1).unwrap();
        assert_eq!(caches.len(), 3);
        assert_eq!(middle.key_plane(), 2);
        assert_eq!(middle.value_plane(), 3);
        assert!(caches.layer(3).is_err());
    }
}
