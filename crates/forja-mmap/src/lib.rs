//! Read-only mappings for operator-owned immutable files.

use std::{fmt, fs::File, io, sync::Arc};

use memmap2::Mmap;

/// A shared read-only mapping of an operator-owned immutable file.
///
/// The file must not be truncated or modified while this value or any clone exists. Forja's
/// grant boundary assigns that responsibility to the operator; violating it can change bytes
/// during inference or terminate the process with a bus error.
#[derive(Clone)]
pub struct MappedRegion {
    mapping: Arc<Mmap>,
    offset: usize,
    len: usize,
}

impl MappedRegion {
    /// Maps an entire file with read-only access.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be mapped.
    pub fn map(file: &File) -> io::Result<Self> {
        // SAFETY: The trusted host only passes operator-owned grant files whose documented
        // contract forbids truncation or mutation for the lifetime of the mapping.
        let mapping = unsafe { Mmap::map(file) }?;
        let len = mapping.len();
        Ok(Self {
            mapping: Arc::new(mapping),
            offset: 0,
            len,
        })
    }

    /// Returns a view beginning at `offset` while retaining the full file mapping.
    ///
    /// # Errors
    ///
    /// Returns an error when `offset` lies outside this region.
    pub fn split_at(self, offset: usize) -> io::Result<Self> {
        if offset > self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "mapped region offset exceeds its length",
            ));
        }
        let start = self.offset.checked_add(offset).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "mapped region offset overflowed",
            )
        })?;
        Ok(Self {
            mapping: self.mapping,
            offset: start,
            len: self.len - offset,
        })
    }

    /// Returns the mapped bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.mapping[self.offset..self.offset + self.len]
    }

    /// Returns the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Reports whether the mapping contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a pointer to the page-aligned mapping base.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.mapping.as_ptr()
    }

    /// Returns the byte displacement from the page-aligned mapping base.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the full page-aligned mapping length.
    #[must_use]
    pub fn mapped_len(&self) -> usize {
        self.mapping.len()
    }
}

impl fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedRegion")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
