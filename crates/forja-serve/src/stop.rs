use std::str::Utf8Error;

/// Incrementally matches UTF-8 stop strings without leaking partial matches.
pub struct StopMatcher {
    stops: Vec<Vec<u8>>,
    pending: Vec<u8>,
    matched: Option<usize>,
}

/// Text released by one stop-matcher update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StopOutput {
    /// Complete text safe to emit.
    pub text: String,
    /// Whether a stop string matched.
    pub matched: bool,
}

impl StopMatcher {
    /// Creates a matcher for nonempty stop strings.
    #[must_use]
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops
                .iter()
                .filter(|stop| !stop.is_empty())
                .map(|stop| stop.as_bytes().to_vec())
                .collect(),
            pending: Vec::new(),
            matched: None,
        }
    }

    /// Returns the index of the matched stop, if any.
    #[must_use]
    pub const fn matched_index(&self) -> Option<usize> {
        self.matched
    }

    /// Adds decoded bytes and returns text that cannot become part of a stop.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes cannot form UTF-8.
    pub fn push(&mut self, bytes: &[u8]) -> Result<StopOutput, Utf8Error> {
        if self.matched.is_some() {
            return Ok(StopOutput {
                text: String::new(),
                matched: true,
            });
        }
        self.pending.extend_from_slice(bytes);
        for start in 0..self.pending.len() {
            let candidate = &self.pending[start..];
            if let Some(index) = self
                .stops
                .iter()
                .position(|stop| candidate.starts_with(stop))
            {
                let text = std::str::from_utf8(&self.pending[..start])?.to_owned();
                self.pending.clear();
                self.matched = Some(index);
                return Ok(StopOutput {
                    text,
                    matched: true,
                });
            }
            if self.stops.iter().any(|stop| stop.starts_with(candidate)) {
                let text = std::str::from_utf8(&self.pending[..start])?.to_owned();
                self.pending.drain(..start);
                return Ok(StopOutput {
                    text,
                    matched: false,
                });
            }
        }
        let emit = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(error) => return Err(error),
        };
        let text = std::str::from_utf8(&self.pending[..emit])?.to_owned();
        self.pending.drain(..emit);
        Ok(StopOutput {
            text,
            matched: false,
        })
    }

    /// Releases buffered text after the token stream ends.
    ///
    /// # Errors
    ///
    /// Returns an error when the final bytes are incomplete UTF-8.
    pub fn finish(&mut self) -> Result<StopOutput, Utf8Error> {
        if self.matched.is_some() {
            return Ok(StopOutput {
                text: String::new(),
                matched: true,
            });
        }
        if let Some((start, index)) = (0..self.pending.len()).find_map(|start| {
            self.stops
                .iter()
                .position(|stop| self.pending[start..].starts_with(stop))
                .map(|index| (start, index))
        }) {
            let text = std::str::from_utf8(&self.pending[..start])?.to_owned();
            self.pending.clear();
            self.matched = Some(index);
            return Ok(StopOutput {
                text,
                matched: true,
            });
        }
        let text = std::str::from_utf8(&self.pending)?.to_owned();
        self.pending.clear();
        Ok(StopOutput {
            text,
            matched: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(stops: &[&str]) -> StopMatcher {
        StopMatcher::new(
            &stops
                .iter()
                .map(|stop| (*stop).to_owned())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn holds_stop_prefixes_across_chunks() {
        let mut matcher = matcher(&["END"]);
        assert_eq!(matcher.push(b"hello E").unwrap().text, "hello ");
        assert_eq!(matcher.push(b"N").unwrap().text, "");
        assert_eq!(
            matcher.push(b"Dignored").unwrap(),
            StopOutput {
                text: String::new(),
                matched: true
            }
        );
    }

    #[test]
    fn matches_a_stop_on_the_first_push() {
        let mut matcher = matcher(&["END"]);
        assert_eq!(
            matcher.push(b"END").unwrap(),
            StopOutput {
                text: String::new(),
                matched: true
            }
        );
    }

    #[test]
    fn earliest_candidate_wins_with_overlapping_stops() {
        let mut matcher = matcher(&["abc", "b"]);
        assert_eq!(matcher.push(b"ab").unwrap().text, "");
        assert_eq!(
            matcher.push(b"x").unwrap(),
            StopOutput {
                text: "a".to_owned(),
                matched: true
            }
        );
        assert_eq!(matcher.matched_index(), Some(1));
    }

    #[test]
    fn handles_split_multibyte_characters() {
        let snowman = "☃".as_bytes();
        let mut stop_matcher = matcher(&["☃!"]);
        assert_eq!(stop_matcher.push(&snowman[..2]).unwrap().text, "");
        assert_eq!(stop_matcher.push(&snowman[2..]).unwrap().text, "");
        assert!(stop_matcher.push(b"!").unwrap().matched);

        let accent = "é".as_bytes();
        let mut passthrough = matcher(&[]);
        assert_eq!(passthrough.push(&accent[..1]).unwrap().text, "");
        assert_eq!(passthrough.push(&accent[1..]).unwrap().text, "é");
    }

    #[test]
    fn finish_releases_unmatched_holdback() {
        let mut matcher = matcher(&["END"]);
        assert_eq!(matcher.push(b"the E").unwrap().text, "the ");
        assert_eq!(matcher.finish().unwrap().text, "E");
    }

    #[test]
    fn finish_matches_a_complete_stop_behind_a_longer_prefix() {
        let mut matcher = matcher(&["abc", "b"]);
        assert_eq!(matcher.push(b"ab").unwrap().text, "");
        assert_eq!(
            matcher.finish().unwrap(),
            StopOutput {
                text: "a".to_owned(),
                matched: true
            }
        );
    }

    #[test]
    fn finish_releases_a_split_multibyte_stop_prefix() {
        let snowman = "☃".as_bytes();
        let mut matcher = matcher(&["☃!"]);
        assert_eq!(matcher.push(&snowman[..2]).unwrap().text, "");
        assert_eq!(matcher.push(&snowman[2..]).unwrap().text, "");
        assert_eq!(
            matcher.finish().unwrap(),
            StopOutput {
                text: "☃".to_owned(),
                matched: false
            }
        );
    }
}
