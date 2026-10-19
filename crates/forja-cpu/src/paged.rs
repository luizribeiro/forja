//! Reference execution for host-owned paged storage.

use forja_core::{BackendError, DType, PageTable, PagedLayout};

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{PageTableEntry, PagedPlane};

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
}
