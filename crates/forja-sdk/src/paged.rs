use std::marker::PhantomData;

use crate::{Dim, Element, Error, Result, Tensor};

/// A borrowed capability for the host-owned paged-storage pool.
pub struct PagedPool<'a> {
    #[cfg(target_family = "wasm")]
    raw: &'a crate::sys::guest::paged_storage::PagePool,
    lifetime: PhantomData<&'a ()>,
}

/// A borrowed capability for one host-filled serving slot.
pub struct PagedSlot<'a> {
    #[cfg(target_family = "wasm")]
    pub(crate) raw: &'a crate::sys::guest::paged_storage::BatchSlot,
    lifetime: PhantomData<&'a ()>,
}

/// Geometry of one private, three-version serving state plane.
pub struct PagedStateDescriptor {
    /// Floating-point storage type.
    pub dtype: crate::DType,
    /// Per-sequence plane dimensions.
    pub shape: Vec<u32>,
}

/// Host-filled lane metadata used by serving graphs.
pub struct PagedSlotTensors {
    /// Per-lane logical sequence lengths.
    pub logical_lengths: Tensor<u32>,
    /// Per-lane query starts.
    pub query_starts: Tensor<u32>,
    /// Per-lane absolute sampling positions.
    pub absolute_positions: Tensor<u32>,
}

impl PagedSlot<'_> {
    #[cfg(target_family = "wasm")]
    #[doc(hidden)]
    #[must_use]
    pub const fn from_guest(raw: &crate::sys::guest::paged_storage::BatchSlot) -> PagedSlot<'_> {
        PagedSlot {
            raw,
            lifetime: PhantomData,
        }
    }

    /// Returns device-visible aliases of host-owned lane metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when the host refuses the slot or its lane count.
    #[cfg(target_family = "wasm")]
    pub fn tensors(&self) -> Result<PagedSlotTensors> {
        let tensors = self
            .raw
            .tensors()
            .map_err(|error| Error::loading(format!("{error:?}")))?;
        let shape = vec![tensors.lanes];
        Ok(PagedSlotTensors {
            logical_lengths: Tensor::from_guest(tensors.logical_lengths, shape.clone()),
            query_starts: Tensor::from_guest(tensors.query_starts, shape.clone()),
            absolute_positions: Tensor::from_guest(tensors.absolute_positions, shape),
        })
    }
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
        self.cache_set_with_state(layers, &[])
    }

    /// Allocates attention caches and opaque state planes within the same serving slot.
    ///
    /// # Errors
    /// Refuses invalid geometry, overflow or insufficient host quota.
    pub fn cache_set_with_state(
        self,
        layers: u32,
        state: &[PagedStateDescriptor],
    ) -> Result<PagedCacheSet> {
        let planes = layers
            .checked_mul(2)
            .ok_or_else(|| Error::loading("paged cache plane count overflowed"))?;
        if planes == 0 {
            return Err(Error::loading("paged cache requires at least one layer"));
        }
        #[cfg(target_family = "wasm")]
        let slot = if state.is_empty() {
            self.raw.open_slot(planes)
        } else {
            let descriptors = state
                .iter()
                .map(
                    |plane| crate::sys::guest::paged_storage::StatePlaneDescriptor {
                        dtype: crate::sys::guest::wit_dtype(plane.dtype),
                        shape: plane.shape.clone(),
                    },
                )
                .collect::<Vec<_>>();
            self.raw.open_state_slot(planes, &descriptors)
        }
        .map_err(|error| Error::loading(format!("{error:?}")))?;
        #[cfg(not(target_family = "wasm"))]
        let _ = state;
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
        self.layer_at(0, layer)
    }

    /// Borrows one layer's key/value planes for a serving lane.
    ///
    /// # Errors
    ///
    /// Returns an error when `layer` is outside this set.
    pub fn layer_at(&self, lane: u32, layer: u32) -> Result<PagedCache<'_>> {
        if layer >= self.layers {
            return Err(Error::loading("paged cache layer is out of range"));
        }
        let key_plane = layer
            .checked_mul(2)
            .ok_or_else(|| Error::loading("paged cache plane index overflowed"))?;
        Ok(PagedCache {
            _set: self,
            lane,
            key_plane,
            value_plane: key_plane + 1,
        })
    }
}

/// A borrowed key/value plane pair for one transformer layer.
pub struct PagedCache<'a> {
    _set: &'a PagedCacheSet,
    lane: u32,
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

    /// Returns the host slot lane used by this cache pair.
    #[must_use]
    pub const fn lane(&self) -> u32 {
        self.lane
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
        let [tokens, _, _]: [u32; 3] = key
            .shape()
            .try_into()
            .map_err(|_| Error::loading("paged key must have rank three"))?;
        if value.shape() != key.shape() {
            return Err(Error::loading("paged key and value shapes differ"));
        }
        #[cfg(target_family = "wasm")]
        let start = crate::graph::affine(start)?;
        #[cfg(target_family = "wasm")]
        crate::graph::record_import(|commands| {
            crate::sys::paged_copy(
                commands,
                &self._set._slot,
                self.lane,
                self.key_plane,
                start,
                u64::from(tokens),
                key.handle(),
            )?;
            crate::sys::paged_copy(
                commands,
                &self._set._slot,
                self.lane,
                self.value_plane,
                start,
                u64::from(tokens),
                value.handle(),
            )
        })?;
        #[cfg(not(target_family = "wasm"))]
        let _ = (&key, &value, start, tokens);
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
    pub fn attention<'a, T: Element>(
        &self,
        query: &Tensor<T>,
        scale: f32,
        start: &Dim,
        end: &Dim,
        selection: impl Into<crate::target::metal::VariantChoice<'a>>,
    ) -> Result<Tensor<T>> {
        self.record_attention(query, None, scale, start, end, selection.into())
    }

    /// Appends one key/value row and attends through the opaque page capability.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible single-row shapes or refused host work.
    pub fn append_attention<'a, T: Element>(
        &self,
        query: &Tensor<T>,
        appended: [&Tensor<T>; 2],
        scale: f32,
        start: &Dim,
        end: &Dim,
        selection: impl Into<crate::target::metal::VariantChoice<'a>>,
    ) -> Result<Tensor<T>> {
        let [key, value] = appended;
        if query.shape().len() != 3
            || query.shape()[0] != 1
            || key.shape().len() != 3
            || key.shape()[0] != 1
            || key.shape() != value.shape()
            || key.shape()[2] != query.shape()[2]
        {
            return Err(Error::loading(
                "paged append attention requires compatible single-row shapes",
            ));
        }
        self.record_attention(query, Some(appended), scale, start, end, selection.into())
    }

    fn record_attention<T: Element>(
        &self,
        query: &Tensor<T>,
        appended: Option<[&Tensor<T>; 2]>,
        scale: f32,
        start: &Dim,
        end: &Dim,
        selection: crate::target::metal::VariantChoice<'_>,
    ) -> Result<Tensor<T>> {
        let [sequence, heads, width]: [u32; 3] = query
            .shape()
            .try_into()
            .map_err(|_| Error::loading("paged attention query must have rank three"))?;
        let hidden = heads
            .checked_mul(width)
            .ok_or_else(|| Error::loading("paged attention width overflowed"))?;
        let query =
            bind_replay_start_to_permuted_query(query.permute(&[1, 0, 2])?, start, sequence)?;
        let output = Tensor::zeros(&[sequence, heads, width])?;
        let paged_output = output.permute(&[1, 0, 2])?;
        #[cfg(target_family = "wasm")]
        let (start, end) = (crate::graph::affine(start)?, crate::graph::affine(end)?);
        #[cfg(target_family = "wasm")]
        crate::graph::record_import_variant(
            selection,
            |commands, name| {
                crate::sys::paged_sdpa(
                    commands,
                    &self._set._slot,
                    self.lane,
                    query.handle(),
                    appended.map(|rows| rows.map(Tensor::handle)),
                    paged_output.handle(),
                    self.key_plane,
                    self.value_plane,
                    scale,
                    true,
                    start,
                    end,
                    name,
                )
            },
            |commands, params, parameter, arms| {
                crate::sys::paged_sdpa(
                    commands,
                    &self._set._slot,
                    self.lane,
                    query.handle(),
                    appended.map(|rows| rows.map(Tensor::handle)),
                    paged_output.handle(),
                    self.key_plane,
                    self.value_plane,
                    scale,
                    true,
                    start,
                    end,
                    "",
                )?;
                crate::sys::select_paged_sdpa_variant_rule(commands, params, parameter, arms)
            },
        )?;
        #[cfg(not(target_family = "wasm"))]
        let _ = (
            self,
            &query,
            appended,
            &paged_output,
            scale,
            start,
            end,
            selection,
        );
        output.reshape(&[sequence, hidden])
    }
}

fn bind_replay_start_to_permuted_query<T: Element>(
    query: Tensor<T>,
    start: &Dim,
    sequence: u32,
) -> Result<Tensor<T>> {
    if start.is_symbolic() {
        query.narrow(1, start.clone().checked_mul(0)?, sequence)
    } else {
        Ok(query)
    }
}

#[cfg(test)]
mod tests {
    use super::PagedCacheSet;

    #[cfg(feature = "native")]
    #[test]
    fn append_attention_requires_compatible_single_rows() {
        use crate::{Dim, Tensor};
        let caches = PagedCacheSet { layers: 1 };
        let cache = caches.layer(0).unwrap();
        let query = Tensor::<f32>::zeros(&[1, 16, 128]).unwrap();
        let key = Tensor::<f32>::zeros(&[1, 8, 128]).unwrap();
        let variant =
            crate::target::metal::Variant::new("paged-sdpa-append.vector-gqa-two-pass-32").unwrap();
        let start = Dim::from(64_u32);
        let end = Dim::from(65_u32);
        assert_eq!(
            cache
                .append_attention(&query, [&key, &key], 0.125, &start, &end, &variant)
                .unwrap()
                .shape(),
            &[1, 2048]
        );
        for shape in [&[7, 8, 128][..], &[1, 8, 64], &[1024]] {
            let bad = Tensor::<f32>::zeros(shape).unwrap();
            assert!(
                cache
                    .append_attention(&query, [&bad, &bad], 0.125, &start, &end, &variant)
                    .is_err()
            );
        }
    }

    #[test]
    fn assigns_one_key_value_pair_per_layer() {
        let caches = PagedCacheSet { layers: 3 };
        let middle = caches.layer(1).unwrap();
        let batched = caches.layer_at(7, 2).unwrap();
        assert_eq!(caches.len(), 3);
        assert_eq!(middle.lane(), 0);
        assert_eq!(middle.key_plane(), 2);
        assert_eq!(middle.value_plane(), 3);
        assert_eq!(batched.lane(), 7);
        assert_eq!(batched.key_plane(), 4);
        assert!(caches.layer(3).is_err());
    }
}
