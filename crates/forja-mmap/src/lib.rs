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
        Ok(Self {
            mapping: Arc::new(mapping),
        })
    }

    /// Returns the mapped bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.mapping
    }

    /// Returns the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Reports whether the mapping contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Returns a pointer to the first mapped byte.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.mapping.as_ptr()
    }
}

impl fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedRegion")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
