//! Distinct representations for constrained token selection.

/// Packed allow words, with bit `id % 32` of word `id / 32` permitting a token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedMask<'a>(&'a [u32]);

impl<'a> PackedMask<'a> {
    /// Borrows words already packed by the caller.
    #[must_use]
    pub const fn new(words: &'a [u32]) -> Self {
        Self(words)
    }

    /// Returns the packed words for kernel or component serialization.
    #[must_use]
    pub const fn words(self) -> &'a [u32] {
        self.0
    }
}

/// Token ids for lane selection; the receiving engine validates ordering and vocabulary.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenIds(Vec<u32>);

impl TokenIds {
    /// Owns a list of token ids rather than packed words.
    #[must_use]
    pub const fn new(ids: Vec<u32>) -> Self {
        Self(ids)
    }

    /// Borrows the token ids for validation or mask construction.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Returns the token ids for component serialization.
    #[must_use]
    pub fn into_vec(self) -> Vec<u32> {
        self.0
    }
}
