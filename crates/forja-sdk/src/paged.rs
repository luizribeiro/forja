use std::marker::PhantomData;

use crate::{Error, Result};

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
