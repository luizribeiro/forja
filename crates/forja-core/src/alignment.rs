/// Natural vector alignment of a byte address, capped at sixteen bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ByteAlignment(u8);

impl ByteAlignment {
    /// Classifies an absolute byte address or an allocation-relative byte start.
    /// A relative classification is valid only with a sixteen-byte-aligned allocation origin.
    #[must_use]
    pub const fn of(address: u64) -> Self {
        let zeros = address.trailing_zeros();
        Self(if zeros >= 4 { 16 } else { 1 << zeros })
    }

    /// Reports whether this class satisfies a natural vector alignment.
    #[must_use]
    pub const fn supports(self, bytes: u64) -> bool {
        bytes.is_power_of_two() && bytes <= self.0 as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_absolute_origins_and_subviews_exhaustively() {
        for base in 0..32 {
            for offset in 0..32 {
                for width in [1, 2, 4, 8, 16] {
                    let bound = ByteAlignment::of(base + offset);
                    assert_eq!(bound.supports(width), (base + offset) % width == 0);
                }
            }
        }
        assert!(!ByteAlignment::of(0).supports(0));
        assert!(!ByteAlignment::of(0).supports(3));
        assert!(!ByteAlignment::of(0).supports(32));
        assert_eq!(ByteAlignment::of(u64::MAX), ByteAlignment::of(1));
    }
}
