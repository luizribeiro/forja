use std::ops::Range;

/// Dense byte regions initialized by earlier commands in a validated submission order.
#[derive(Default)]
pub struct WrittenRanges(Vec<Range<u64>>);

impl WrittenRanges {
    /// Adds a dense written region, merging overlapping or adjacent spans.
    pub fn record(&mut self, mut range: Range<u64>) {
        if range.start >= range.end {
            return;
        }
        let start = self.0.partition_point(|prior| prior.end < range.start);
        let mut end = start;
        while self
            .0
            .get(end)
            .is_some_and(|prior| prior.start <= range.end)
        {
            range.start = range.start.min(self.0[end].start);
            range.end = range.end.max(self.0[end].end);
            end += 1;
        }
        self.0.splice(start..end, [range]);
    }

    /// Reports whether every byte in a requested span has been initialized.
    #[must_use]
    pub fn covers(&self, range: &Range<u64>) -> bool {
        range.start >= range.end
            || self
                .0
                .get(self.0.partition_point(|prior| prior.end < range.end))
                .is_some_and(|prior| prior.start <= range.start && range.end <= prior.end)
    }

    /// Returns the portions of a span that still depend on older contents.
    #[must_use]
    pub fn uncovered(&self, mut range: Range<u64>) -> Vec<Range<u64>> {
        let mut result = Vec::new();
        for written in &self.0 {
            if written.end <= range.start {
                continue;
            }
            if written.start >= range.end {
                break;
            }
            if range.start < written.start {
                result.push(range.start..written.start);
            }
            range.start = range.start.max(written.end);
            if range.start >= range.end {
                break;
            }
        }
        if range.start < range.end {
            result.push(range);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_out_of_order_writes_without_covering_holes() {
        let mut written = WrittenRanges::default();
        for range in [33..64, 0..7, 7..17, 60..65] {
            written.record(range);
        }
        assert!(written.covers(&(0..17)));
        assert!(written.covers(&(34..63)));
        assert!(!written.covers(&(16..34)));
        assert_eq!(written.uncovered(0..70), [17..33, 65..70]);
        written.record(17..33);
        assert!(written.covers(&(0..65)));
        assert_eq!(written.uncovered(1..64), []);
    }

    #[test]
    fn handles_empty_nested_and_maximum_extents() {
        let mut written = WrittenRanges::default();
        written.record(7..7);
        assert_eq!(written.uncovered(0..7), std::slice::from_ref(&(0..7)));
        written.record(7..u64::MAX);
        written.record(33..4097);
        assert!(written.covers(&(u64::MAX - 1..u64::MAX)));
        assert!(!written.covers(&(0..u64::MAX)));
        assert_eq!(
            written.uncovered(0..u64::MAX),
            std::slice::from_ref(&(0..7))
        );
        written.record(0..7);
        assert!(written.covers(&(0..u64::MAX)));
    }
}
