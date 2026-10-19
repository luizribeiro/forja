//! Reference execution for host-owned paged storage.

use forja_core::{BackendError, DType, PageTable, PagedLayout};
use half::{bf16, f16};

/// CPU backing storage for a validated paged layout.
#[derive(Debug)]
pub struct CpuPagePool {
    layout: PagedLayout,
    physical_pages: u32,
    planes: Vec<Vec<u8>>,
}

impl CpuPagePool {
    /// Allocates zeroed backing for every plane and physical page.
    ///
    /// # Errors
    ///
    /// Returns an allocation error when a byte count overflows or memory cannot be reserved.
    pub fn new(layout: PagedLayout, physical_pages: u32) -> Result<Self, BackendError> {
        if physical_pages == 0 {
            return Err(BackendError::InvalidInput);
        }
        let mut planes = Vec::with_capacity(layout.planes().len());
        for plane in 0..layout.planes().len() {
            let bytes = layout
                .plane_page_bytes(plane)
                .and_then(|page| page.checked_mul(u64::from(physical_pages)))
                .ok_or(BackendError::AllocationFailed)?;
            let len = usize::try_from(bytes).map_err(|_| BackendError::AllocationFailed)?;
            let mut storage = Vec::new();
            storage
                .try_reserve_exact(len)
                .map_err(|_| BackendError::AllocationFailed)?;
            storage.resize(len, 0);
            planes.push(storage);
        }
        Ok(Self {
            layout,
            physical_pages,
            planes,
        })
    }

    /// Creates CPU backing from trusted, exactly sized plane bytes.
    #[doc(hidden)]
    pub fn from_plane_bytes(
        layout: PagedLayout,
        physical_pages: u32,
        planes: Vec<Vec<u8>>,
    ) -> Result<Self, BackendError> {
        if physical_pages == 0 || planes.len() != layout.planes().len() {
            return Err(BackendError::InvalidInput);
        }
        for (plane, bytes) in planes.iter().enumerate() {
            let expected = layout
                .plane_page_bytes(plane)
                .and_then(|page| page.checked_mul(u64::from(physical_pages)))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or(BackendError::InvalidInput)?;
            if bytes.len() != expected {
                return Err(BackendError::InvalidInput);
            }
        }
        Ok(Self {
            layout,
            physical_pages,
            planes,
        })
    }

    /// Copies contiguous token rows into a logical paged span.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a bad plane, table entry, range, or byte length.
    pub fn copy_in(
        &mut self,
        plane: usize,
        table: &PageTable,
        start: u64,
        tokens: u64,
        source: &[u8],
    ) -> Result<(), BackendError> {
        let row_bytes = self.row_bytes(plane)?;
        let expected = tokens
            .checked_mul(row_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(BackendError::InvalidInput)?;
        if source.len() != expected {
            return Err(BackendError::InvalidInput);
        }
        for token in 0..tokens {
            let logical = start.checked_add(token).ok_or(BackendError::InvalidInput)?;
            let destination = self.token_range(plane, table, logical)?;
            let source_start = usize::try_from(
                token
                    .checked_mul(row_bytes)
                    .ok_or(BackendError::InvalidInput)?,
            )
            .map_err(|_| BackendError::InvalidInput)?;
            let source_end = source_start
                .checked_add(usize::try_from(row_bytes).map_err(|_| BackendError::InvalidInput)?)
                .ok_or(BackendError::InvalidInput)?;
            self.planes
                .get_mut(plane)
                .and_then(|storage| storage.get_mut(destination))
                .ok_or(BackendError::InvalidInput)?
                .copy_from_slice(
                    source
                        .get(source_start..source_end)
                        .ok_or(BackendError::InvalidInput)?,
                );
        }
        Ok(())
    }

    /// Copies a logical paged span into contiguous token rows.
    ///
    /// # Errors
    ///
    /// Returns invalid input for a bad plane, table entry, or range.
    pub fn copy_out(
        &self,
        plane: usize,
        table: &PageTable,
        start: u64,
        tokens: u64,
    ) -> Result<Vec<u8>, BackendError> {
        let row_bytes = self.row_bytes(plane)?;
        let byte_len = tokens
            .checked_mul(row_bytes)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(BackendError::InvalidInput)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(byte_len)
            .map_err(|_| BackendError::AllocationFailed)?;
        for token in 0..tokens {
            let logical = start.checked_add(token).ok_or(BackendError::InvalidInput)?;
            let range = self.token_range(plane, table, logical)?;
            output.extend_from_slice(
                self.planes
                    .get(plane)
                    .and_then(|storage| storage.get(range))
                    .ok_or(BackendError::InvalidInput)?,
            );
        }
        Ok(output)
    }

    /// Computes grouped-query SDPA directly over logical K/V page spans.
    ///
    /// Query storage is f32 in `[head, query, width]` order. The output uses the same order with
    /// `value_width` as its innermost extent.
    ///
    /// # Errors
    ///
    /// Returns invalid input for inconsistent dimensions, ranges, or page entries.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa(
        &self,
        table: &PageTable,
        key_plane: usize,
        value_plane: usize,
        query: &[f32],
        query_heads: u32,
        query_length: u32,
        kv_heads: u32,
        width: u32,
        value_width: u32,
        key_length: u32,
        query_start: u32,
        scale: f32,
        causal: bool,
    ) -> Result<Vec<f32>, BackendError> {
        let expected_query = checked_product(&[query_heads, query_length, width])?;
        let key_elements = u64::from(kv_heads)
            .checked_mul(u64::from(width))
            .ok_or(BackendError::InvalidInput)?;
        let value_elements = u64::from(kv_heads)
            .checked_mul(u64::from(value_width))
            .ok_or(BackendError::InvalidInput)?;
        if query.len() != expected_query
            || query_heads == 0
            || kv_heads == 0
            || !query_heads.is_multiple_of(kv_heads)
            || self.plane_elements(key_plane)? != key_elements
            || self.plane_elements(value_plane)? != value_elements
            || !scale.is_finite()
        {
            return Err(BackendError::InvalidInput);
        }
        let output_len = checked_product(&[query_heads, query_length, value_width])?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| BackendError::AllocationFailed)?;
        let heads_per_group = query_heads / kv_heads;
        for head in 0..query_heads {
            let kv_head = head / heads_per_group;
            for query_row in 0..query_length {
                let query_offset = usize::try_from(head)
                    .ok()
                    .and_then(|head| {
                        head.checked_mul(checked_product(&[query_length, width]).ok()?)
                    })
                    .and_then(|offset| {
                        offset.checked_add(checked_product(&[query_row, width]).ok()?)
                    })
                    .ok_or(BackendError::InvalidInput)?;
                let query_end = query_offset
                    .checked_add(usize::try_from(width).map_err(|_| BackendError::InvalidInput)?)
                    .ok_or(BackendError::InvalidInput)?;
                let query_row_values = query
                    .get(query_offset..query_end)
                    .ok_or(BackendError::InvalidInput)?;
                let attended = if causal {
                    key_length.min(
                        query_start
                            .checked_add(query_row)
                            .and_then(|position| position.checked_add(1))
                            .ok_or(BackendError::InvalidInput)?,
                    )
                } else {
                    key_length
                };
                let mut scores = Vec::with_capacity(
                    usize::try_from(attended).map_err(|_| BackendError::InvalidInput)?,
                );
                for key_row in 0..attended {
                    let mut score = 0.0_f32;
                    for column in 0..width {
                        let element = row_element(kv_head, width, column)?;
                        score += query_row_values
                            [usize::try_from(column).map_err(|_| BackendError::InvalidInput)?]
                            * self.load(key_plane, table, key_row, element)?;
                    }
                    scores.push(score * scale);
                }
                softmax(&mut scores);
                for column in 0..value_width {
                    let mut sum = 0.0_f32;
                    for (key_row, weight) in scores.iter().enumerate() {
                        let element = row_element(kv_head, value_width, column)?;
                        sum += weight
                            * self.load(
                                value_plane,
                                table,
                                u32::try_from(key_row).map_err(|_| BackendError::InvalidInput)?,
                                element,
                            )?;
                    }
                    output.push(sum);
                }
            }
        }
        Ok(output)
    }

    fn row_bytes(&self, plane: usize) -> Result<u64, BackendError> {
        self.plane_elements(plane)?
            .checked_mul(self.plane_dtype(plane)?.byte_size())
            .ok_or(BackendError::InvalidInput)
    }

    fn plane_elements(&self, plane: usize) -> Result<u64, BackendError> {
        self.layout
            .planes()
            .get(plane)
            .map(|descriptor| descriptor.elements_per_token)
            .ok_or(BackendError::InvalidInput)
    }

    fn plane_dtype(&self, plane: usize) -> Result<DType, BackendError> {
        self.layout
            .planes()
            .get(plane)
            .map(|descriptor| descriptor.dtype)
            .ok_or(BackendError::InvalidInput)
    }

    fn token_range(
        &self,
        plane: usize,
        table: &PageTable,
        logical_token: u64,
    ) -> Result<std::ops::Range<usize>, BackendError> {
        let page_tokens = u64::from(self.layout.page_tokens());
        let logical_page = logical_token / page_tokens;
        let physical_page = table
            .resolve(u32::try_from(logical_page).map_err(|_| BackendError::InvalidInput)?)
            .map_err(|_| BackendError::InvalidInput)?;
        if physical_page >= self.physical_pages {
            return Err(BackendError::InvalidInput);
        }
        let page_bytes = self
            .layout
            .plane_page_bytes(plane)
            .ok_or(BackendError::InvalidInput)?;
        let row_bytes = self.row_bytes(plane)?;
        let start = u64::from(physical_page)
            .checked_mul(page_bytes)
            .and_then(|offset| {
                offset.checked_add((logical_token % page_tokens).checked_mul(row_bytes)?)
            })
            .ok_or(BackendError::InvalidInput)?;
        let end = start
            .checked_add(row_bytes)
            .ok_or(BackendError::InvalidInput)?;
        Ok(
            usize::try_from(start).map_err(|_| BackendError::InvalidInput)?
                ..usize::try_from(end).map_err(|_| BackendError::InvalidInput)?,
        )
    }

    fn load(
        &self,
        plane: usize,
        table: &PageTable,
        token: u32,
        element: u64,
    ) -> Result<f32, BackendError> {
        if element >= self.plane_elements(plane)? {
            return Err(BackendError::InvalidInput);
        }
        let token_range = self.token_range(plane, table, u64::from(token))?;
        let dtype = self.plane_dtype(plane)?;
        let offset = token_range
            .start
            .checked_add(
                usize::try_from(
                    element
                        .checked_mul(dtype.byte_size())
                        .ok_or(BackendError::InvalidInput)?,
                )
                .map_err(|_| BackendError::InvalidInput)?,
            )
            .ok_or(BackendError::InvalidInput)?;
        let bytes = self.planes.get(plane).ok_or(BackendError::InvalidInput)?;
        match dtype {
            DType::F32 => read_array::<4>(bytes, offset).map(f32::from_le_bytes),
            DType::F16 => read_array::<2>(bytes, offset)
                .map(u16::from_le_bytes)
                .map(f16::from_bits)
                .map(f32::from),
            DType::BF16 => read_array::<2>(bytes, offset)
                .map(u16::from_le_bytes)
                .map(bf16::from_bits)
                .map(f32::from),
            DType::I32 | DType::U32 => Err(BackendError::InvalidInput),
        }
    }
}

fn row_element(head: u32, width: u32, column: u32) -> Result<u64, BackendError> {
    u64::from(head)
        .checked_mul(u64::from(width))
        .and_then(|offset| offset.checked_add(u64::from(column)))
        .ok_or(BackendError::InvalidInput)
}

fn checked_product(extents: &[u32]) -> Result<usize, BackendError> {
    extents.iter().try_fold(1_usize, |product, &extent| {
        product
            .checked_mul(usize::try_from(extent).map_err(|_| BackendError::InvalidInput)?)
            .ok_or(BackendError::InvalidInput)
    })
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], BackendError> {
    bytes
        .get(offset..offset.checked_add(N).ok_or(BackendError::InvalidInput)?)
        .and_then(|value| value.try_into().ok())
        .ok_or(BackendError::InvalidInput)
}

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{PagePool, PageTableEntry, PagedPlane};

    fn layout() -> PagedLayout {
        PagedLayout::new(
            4,
            vec![PagedPlane {
                dtype: DType::F32,
                elements_per_token: 2,
                alignment: 4,
                label: "values".to_owned(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn paged_copy_crosses_nonconsecutive_pages() {
        let mut pool = CpuPagePool::new(layout(), 3).unwrap();
        let table = PageTable::new(
            vec![PageTableEntry::new(2, 1), PageTableEntry::new(0, 1)],
            vec![1, 1, 1],
        )
        .unwrap();
        let values = (0_u32..12).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        pool.copy_in(0, &table, 2, 6, &values).unwrap();
        assert_eq!(pool.copy_out(0, &table, 2, 6).unwrap(), values);
    }

    #[test]
    fn cow_copy_preserves_shared_prefix_bytes() {
        let layout = layout();
        let pool = PagePool::new(layout.clone(), 4, 4_096).unwrap();
        let mut original = pool.reserve(8).unwrap();
        original.commit_write(7).unwrap();
        let mut storage = CpuPagePool::new(layout, 4).unwrap();
        let prefix = (0_u32..14).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        storage
            .copy_in(0, &original.table().unwrap(), 0, 7, &prefix)
            .unwrap();
        let sibling = original.fork(7).unwrap();

        let copy = original.cow_tail().unwrap().unwrap();
        assert_eq!(copy.token_count(), 3);
        let tail = storage
            .copy_out(0, &sibling.table().unwrap(), 4, 3)
            .unwrap();
        storage
            .copy_in(0, &original.table().unwrap(), 4, 3, &tail)
            .unwrap();

        assert_eq!(
            storage
                .copy_out(0, &original.table().unwrap(), 0, 7)
                .unwrap(),
            prefix
        );
        assert_eq!(
            storage
                .copy_out(0, &sibling.table().unwrap(), 0, 7)
                .unwrap(),
            prefix
        );
    }

    #[test]
    fn paged_attention_matches_a_hand_checked_case() {
        let mut pool = CpuPagePool::new(layout(), 2).unwrap();
        let table = PageTable::new(
            vec![PageTableEntry::new(1, 4), PageTableEntry::new(0, 3)],
            vec![3, 4],
        )
        .unwrap();
        let values = [1.0_f32, 0.0, 0.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        pool.copy_in(0, &table, 0, 2, &values).unwrap();
        let output = pool
            .sdpa(&table, 0, 0, &[1.0, 0.0], 1, 1, 1, 2, 2, 2, 0, 1.0, false)
            .unwrap();
        let expected = std::f32::consts::E / (std::f32::consts::E + 1.0);
        assert!((output[0] - expected).abs() < 1.0e-6);
        assert!((output[1] - (1.0 - expected)).abs() < 1.0e-6);
    }
}
