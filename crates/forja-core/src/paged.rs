use std::{error::Error, fmt};

use crate::DType;

/// Largest accepted token count in one page.
pub const MAX_PAGE_TOKENS: u32 = 4_096;
/// Largest accepted plane count in one pool.
pub const MAX_PAGED_PLANES: usize = 4_096;
const MAX_LABEL_BYTES: usize = 64;
const POISON_GENERATION: u32 = u32::MAX;

/// One authenticated logical-to-physical page mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PageTableEntry {
    page: u32,
    generation: u32,
}

impl PageTableEntry {
    /// Creates an entry from trusted allocator state.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(page: u32, generation: u32) -> Self {
        Self { page, generation }
    }

    /// Returns the physical page index.
    #[must_use]
    pub const fn page(self) -> u32 {
        self.page
    }

    /// Returns the allocation generation carried by the entry.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// A reason a page-table lookup was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageTableError {
    /// The logical page index is outside the table.
    LogicalPageOutOfBounds,
    /// The entry names a physical page outside the pool.
    PhysicalPageOutOfBounds,
    /// The entry generation does not authenticate the current allocation.
    StaleGeneration,
}

impl fmt::Display for PageTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid page table: {self:?}")
    }
}

impl Error for PageTableError {}

/// A host-owned, generation-authenticated logical page table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageTable {
    entries: Vec<PageTableEntry>,
    generations: Vec<u32>,
    contiguous: bool,
}

impl PageTable {
    /// Validates every entry against the current physical-page generations.
    ///
    /// # Errors
    ///
    /// Returns [`PageTableError`] when an entry is out of bounds or stale.
    pub fn new(
        entries: Vec<PageTableEntry>,
        generations: Vec<u32>,
    ) -> Result<Self, PageTableError> {
        for entry in &entries {
            validate_entry(*entry, &generations)?;
        }
        let contiguous = entries
            .windows(2)
            .all(|pair| pair[0].page.checked_add(1) == Some(pair[1].page));
        Ok(Self {
            entries,
            generations,
            contiguous,
        })
    }

    /// Constructs an inactive-lane table whose entries must never be dereferenced.
    #[must_use]
    pub fn poisoned(logical_pages: usize) -> Self {
        Self {
            entries: vec![PageTableEntry::new(0, POISON_GENERATION); logical_pages],
            generations: vec![0],
            contiguous: false,
        }
    }

    /// Pads a valid table with entries carrying the impossible generation.
    ///
    /// # Errors
    ///
    /// Returns [`PageTableError::LogicalPageOutOfBounds`] when `logical_pages` is too small.
    pub fn pad_with_poison(&self, logical_pages: usize) -> Result<Self, PageTableError> {
        if logical_pages < self.entries.len() {
            return Err(PageTableError::LogicalPageOutOfBounds);
        }
        let mut entries = self.entries.clone();
        entries.resize(logical_pages, PageTableEntry::new(0, POISON_GENERATION));
        Ok(Self {
            entries,
            generations: self.generations.clone(),
            contiguous: self.contiguous,
        })
    }

    /// Resolves one logical page after rechecking its generation.
    ///
    /// # Errors
    ///
    /// Returns [`PageTableError`] when the logical index, physical index, or generation is bad.
    pub fn resolve(&self, logical_page: u32) -> Result<u32, PageTableError> {
        let entry = self
            .entries
            .get(
                usize::try_from(logical_page)
                    .map_err(|_| PageTableError::LogicalPageOutOfBounds)?,
            )
            .copied()
            .ok_or(PageTableError::LogicalPageOutOfBounds)?;
        validate_entry(entry, &self.generations)?;
        Ok(entry.page)
    }

    /// Returns whether the validator proved that every physical page is consecutive.
    #[must_use]
    pub const fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    /// Returns the authenticated entries for trusted backend upload.
    #[doc(hidden)]
    #[must_use]
    pub fn entries(&self) -> &[PageTableEntry] {
        &self.entries
    }

    /// Returns current generations for trusted backend upload.
    #[doc(hidden)]
    #[must_use]
    pub fn generations(&self) -> &[u32] {
        &self.generations
    }
}

fn validate_entry(entry: PageTableEntry, generations: &[u32]) -> Result<(), PageTableError> {
    let generation = generations
        .get(usize::try_from(entry.page).map_err(|_| PageTableError::PhysicalPageOutOfBounds)?)
        .ok_or(PageTableError::PhysicalPageOutOfBounds)?;
    if *generation != entry.generation {
        return Err(PageTableError::StaleGeneration);
    }
    Ok(())
}

/// One independently typed plane in a paged storage pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedPlane {
    /// Stored scalar type.
    pub dtype: DType,
    /// Number of stored values for each logical token.
    pub elements_per_token: u64,
    /// Required byte alignment for each page of this plane.
    pub alignment: u64,
    /// Engine-local diagnostic name.
    pub label: String,
}

/// A reason paged storage geometry was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PagedLayoutError {
    /// The page token count is zero or exceeds [`MAX_PAGE_TOKENS`].
    InvalidPageTokens,
    /// A pool must contain between one and [`MAX_PAGED_PLANES`] planes.
    InvalidPlaneCount,
    /// A plane contains no values per token.
    EmptyPlane {
        /// Zero-based plane index.
        plane: usize,
    },
    /// A plane alignment is not a power of two compatible with its dtype.
    InvalidAlignment {
        /// Zero-based plane index.
        plane: usize,
    },
    /// A diagnostic label is empty or too long.
    InvalidLabel {
        /// Zero-based plane index.
        plane: usize,
    },
    /// A byte calculation exceeded 64 bits.
    ArithmeticOverflow,
}

impl fmt::Display for PagedLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid paged layout: {self:?}")
    }
}

impl Error for PagedLayoutError {}

/// Validated, model-neutral geometry shared by every page in a pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PagedLayout {
    page_tokens: u32,
    planes: Vec<PagedPlane>,
    page_bytes: Vec<u64>,
    total_page_bytes: u64,
}

impl PagedLayout {
    /// Validates plane geometry and all derived byte counts.
    ///
    /// # Errors
    ///
    /// Returns [`PagedLayoutError`] for invalid fields or checked-arithmetic overflow.
    pub fn new(page_tokens: u32, planes: Vec<PagedPlane>) -> Result<Self, PagedLayoutError> {
        if page_tokens == 0 || page_tokens > MAX_PAGE_TOKENS {
            return Err(PagedLayoutError::InvalidPageTokens);
        }
        if planes.is_empty() || planes.len() > MAX_PAGED_PLANES {
            return Err(PagedLayoutError::InvalidPlaneCount);
        }
        let mut page_bytes = Vec::with_capacity(planes.len());
        let mut total_page_bytes = 0_u64;
        for (plane, descriptor) in planes.iter().enumerate() {
            if descriptor.elements_per_token == 0 {
                return Err(PagedLayoutError::EmptyPlane { plane });
            }
            if !descriptor.alignment.is_power_of_two()
                || descriptor.alignment < descriptor.dtype.byte_size()
                || !descriptor
                    .alignment
                    .is_multiple_of(descriptor.dtype.byte_size())
            {
                return Err(PagedLayoutError::InvalidAlignment { plane });
            }
            if descriptor.label.is_empty() || descriptor.label.len() > MAX_LABEL_BYTES {
                return Err(PagedLayoutError::InvalidLabel { plane });
            }
            let bytes = u64::from(page_tokens)
                .checked_mul(descriptor.elements_per_token)
                .and_then(|elements| elements.checked_mul(descriptor.dtype.byte_size()))
                .ok_or(PagedLayoutError::ArithmeticOverflow)?;
            let padded = align_up(bytes, descriptor.alignment)?;
            total_page_bytes = total_page_bytes
                .checked_add(padded)
                .ok_or(PagedLayoutError::ArithmeticOverflow)?;
            page_bytes.push(padded);
        }
        Ok(Self {
            page_tokens,
            planes,
            page_bytes,
            total_page_bytes,
        })
    }

    /// Returns the logical token capacity of each page.
    #[must_use]
    pub const fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    /// Returns the validated plane descriptors.
    #[must_use]
    pub fn planes(&self) -> &[PagedPlane] {
        &self.planes
    }

    /// Returns the aligned bytes occupied by one page of a plane.
    #[must_use]
    pub fn plane_page_bytes(&self, plane: usize) -> Option<u64> {
        self.page_bytes.get(plane).copied()
    }

    /// Returns the sum of one corresponding page from every plane.
    #[must_use]
    pub const fn total_page_bytes(&self) -> u64 {
        self.total_page_bytes
    }
}

/// `alignment` is a nonzero power of two validated at the layout boundary.
fn align_up(value: u64, alignment: u64) -> Result<u64, PagedLayoutError> {
    value
        .checked_add(alignment - 1)
        .map(|end| end & !(alignment - 1))
        .ok_or(PagedLayoutError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane() -> PagedPlane {
        PagedPlane {
            dtype: DType::F16,
            elements_per_token: 1_024,
            alignment: 256,
            label: "key".to_owned(),
        }
    }

    #[test]
    fn validates_and_aligns_planes() {
        let layout = PagedLayout::new(33, vec![plane()]).unwrap();
        assert_eq!(layout.plane_page_bytes(0), Some(67_584));
        assert_eq!(layout.total_page_bytes(), 67_584);
    }

    #[test]
    fn refuses_zero_and_oversized_pages() {
        assert_eq!(
            PagedLayout::new(0, vec![plane()]),
            Err(PagedLayoutError::InvalidPageTokens)
        );
        assert_eq!(
            PagedLayout::new(MAX_PAGE_TOKENS + 1, vec![plane()]),
            Err(PagedLayoutError::InvalidPageTokens)
        );
    }

    #[test]
    fn refuses_bad_plane_layouts_and_overflow() {
        let mut bad = plane();
        bad.alignment = 3;
        assert_eq!(
            PagedLayout::new(64, vec![bad]),
            Err(PagedLayoutError::InvalidAlignment { plane: 0 })
        );
        let mut overflowing = plane();
        overflowing.elements_per_token = u64::MAX;
        assert_eq!(
            PagedLayout::new(64, vec![overflowing]),
            Err(PagedLayoutError::ArithmeticOverflow)
        );
    }

    #[test]
    fn authenticates_and_classifies_page_tables() {
        let table = PageTable::new(
            vec![PageTableEntry::new(2, 7), PageTableEntry::new(3, 11)],
            vec![0, 0, 7, 11],
        )
        .unwrap();
        assert_eq!(table.resolve(1), Ok(3));
        assert!(table.is_contiguous());
        assert_eq!(
            PageTable::new(vec![PageTableEntry::new(2, 6)], vec![0, 0, 7]),
            Err(PageTableError::StaleGeneration)
        );
    }

    #[test]
    fn poisoned_tables_fail_if_dereferenced() {
        assert_eq!(
            PageTable::poisoned(1).resolve(0),
            Err(PageTableError::StaleGeneration)
        );
    }
}
