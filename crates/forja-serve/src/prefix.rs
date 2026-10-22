use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use forja_core::PagePoolError;
use forja_host::PagedPrefix;
const BLOCK: usize = 64;
const BLOCK_DOMAIN: &[u8] = b"forja-prefix-block-v1";
const NAMESPACE_DOMAIN: &[u8] = b"forja-prefix-namespace-v1";
static RUNNER_NONCE: AtomicU64 = AtomicU64::new(0);

/// Fixed metadata bounds for one shared-prefix index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
pub(crate) struct PrefixLimits {
    pub(crate) entries: usize,
    pub(crate) tokens: usize,
    pub(crate) referenced_pages: usize,
}

/// A shared-prefix index operation failure.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
pub(crate) enum PrefixIndexError {
    InvalidLimits,
    EntryTooLarge,
    Bookkeeping,
    Pool(PagePoolError),
}

impl fmt::Display for PrefixIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits => formatter.write_str("prefix index limits must be nonzero"),
            Self::EntryTooLarge => formatter.write_str("prefix index entry exceeds its limits"),
            Self::Bookkeeping => formatter.write_str("prefix index bookkeeping is inconsistent"),
            Self::Pool(error) => write!(formatter, "prefix index pool error: {error}"),
        }
    }
}

impl Error for PrefixIndexError {}

impl From<PagePoolError> for PrefixIndexError {
    fn from(error: PagePoolError) -> Self {
        Self::Pool(error)
    }
}

/// Artifact identity for one serving engine's live prefix.
pub struct PrefixIdentity {
    fields: [(u64, String); 5],
}

impl PrefixIdentity {
    /// Builds an identity from the engine, profile, weights, tokenizer, and chat-format hashes.
    ///
    /// # Errors
    ///
    /// Returns an error if a hash length cannot be represented as `u64`.
    pub fn new(hashes: [&str; 5]) -> Result<Self, &'static str> {
        let mut fields = [const { (0, String::new()) }; 5];
        for (output, hash) in fields.iter_mut().zip(hashes) {
            output.0 = u64::try_from(hash.len()).map_err(|_| "artifact hash is too long")?;
            hash.clone_into(&mut output.1);
        }
        Ok(Self { fields })
    }

    pub(crate) fn into_namespace(self) -> PrefixNamespace {
        let nonce = RUNNER_NONCE.fetch_add(1, Ordering::Relaxed);
        self.namespace_with_nonce(nonce)
    }

    fn namespace_with_nonce(&self, nonce: u64) -> PrefixNamespace {
        let mut layout = Sha256::new();
        layout.update(self.fields[0].1.as_bytes());
        layout.update(nonce.to_le_bytes());
        let layout: [u8; 32] = layout.finalize().into();
        let mut hash = Sha256::new();
        hash.update(NAMESPACE_DOMAIN);
        for (length, value) in &self.fields {
            hash.update(length.to_le_bytes());
            hash.update(value.as_bytes());
        }
        hash.update(32_u64.to_le_bytes());
        hash.update(layout);
        PrefixNamespace(hash.finalize().into())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PrefixNamespace(pub(crate) [u8; 32]);
#[derive(Debug)]
pub(crate) struct LivePrefix {
    namespace: PrefixNamespace,
    tokens: Vec<u32>,
    chain: Vec<[u8; 32]>,
}

#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
struct PrefixEntry {
    fingerprint: LivePrefix,
    sequence: PagedPrefix,
    pages: usize,
    used: u64,
}

/// A sequence fork attached to the longest exact indexed prefix.
#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
pub(crate) struct PrefixHit {
    pub(crate) sequence: PagedPrefix,
    pub(crate) cached_tokens: usize,
}

/// Multi-entry prefix index retaining immutable page references.
#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
pub(crate) struct PrefixIndex {
    limits: PrefixLimits,
    entries: HashMap<u64, PrefixEntry>,
    blocks: HashMap<[u8; 32], Vec<u64>>,
    next_id: u64,
    clock: u64,
    tokens: usize,
    pages: usize,
}

#[cfg_attr(not(test), allow(dead_code, reason = "used by paged serving"))]
impl PrefixIndex {
    pub(crate) fn new(limits: PrefixLimits) -> Result<Self, PrefixIndexError> {
        if limits.entries == 0 || limits.tokens == 0 || limits.referenced_pages == 0 {
            return Err(PrefixIndexError::InvalidLimits);
        }
        Ok(Self {
            limits,
            entries: HashMap::new(),
            blocks: HashMap::new(),
            next_id: 0,
            clock: 0,
            tokens: 0,
            pages: 0,
        })
    }

    pub(crate) fn publish(
        &mut self,
        namespace: PrefixNamespace,
        tokens: Vec<u32>,
        sequence: &PagedPrefix,
    ) -> Result<(), PrefixIndexError> {
        let token_count = u64::try_from(tokens.len())
            .map_err(|_| PrefixIndexError::Pool(PagePoolError::ArithmeticOverflow))?;
        let pages = loop {
            match sequence.fork_reserved(token_count, token_count) {
                Ok(pages) => break pages,
                Err(PagePoolError::QuotaExceeded) if self.evict_lru(None)? => {}
                Err(PagePoolError::QuotaExceeded) => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        };
        let page_count = pages.page_count()?;
        if tokens.len() > self.limits.tokens || page_count > self.limits.referenced_pages {
            return Err(PrefixIndexError::EntryTooLarge);
        }
        self.evict_for(tokens.len(), page_count)?;
        let id = self.next_id;
        let next_id = self
            .next_id
            .checked_add(1)
            .ok_or(PrefixIndexError::Pool(PagePoolError::ArithmeticOverflow))?;
        let fingerprint = LivePrefix::new(namespace, tokens);
        let token_total = self
            .tokens
            .checked_add(fingerprint.tokens.len())
            .ok_or(PrefixIndexError::Bookkeeping)?;
        let page_total = self
            .pages
            .checked_add(page_count)
            .ok_or(PrefixIndexError::Bookkeeping)?;
        let used = self.tick()?;
        for hash in &fingerprint.chain {
            self.blocks.entry(*hash).or_default().push(id);
        }
        self.next_id = next_id;
        self.tokens = token_total;
        self.pages = page_total;
        self.entries.insert(
            id,
            PrefixEntry {
                fingerprint,
                sequence: pages,
                pages: page_count,
                used,
            },
        );
        Ok(())
    }

    pub(crate) fn lookup(
        &mut self,
        namespace: PrefixNamespace,
        prompt: &[u32],
        reserved_tokens: u64,
    ) -> Result<Option<PrefixHit>, PrefixIndexError> {
        if prompt.is_empty() {
            return Ok(None);
        }
        let cap = prompt.len().saturating_sub(1);
        let prompt_fingerprint = LivePrefix::new(namespace, prompt[..cap].to_vec());
        let mut visited = HashSet::new();
        let mut best = None;
        for block in (0..=prompt_fingerprint.chain.len()).rev() {
            let candidates = block.checked_sub(1).map_or_else(
                || self.entries.keys().copied().collect::<Vec<_>>(),
                |index| {
                    self.blocks
                        .get(&prompt_fingerprint.chain[index])
                        .cloned()
                        .unwrap_or_default()
                },
            );
            for id in candidates {
                if !visited.insert(id) {
                    continue;
                }
                let Some(entry) = self.entries.get(&id) else {
                    continue;
                };
                let matched = entry.fingerprint.match_len(namespace, prompt);
                if best.is_none_or(|(_, length)| matched > length) {
                    best = Some((id, matched));
                }
            }
            if best.is_some_and(|(_, length)| length >= block * BLOCK) {
                break;
            }
        }
        let Some((id, cached_tokens @ 1..)) = best else {
            return Ok(None);
        };
        let used = self.tick()?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or(PrefixIndexError::Pool(PagePoolError::CorruptState))?;
        entry.used = used;
        let cached_tokens_u64 = u64::try_from(cached_tokens)
            .map_err(|_| PrefixIndexError::Pool(PagePoolError::ArithmeticOverflow))?;
        let sequence = loop {
            let result = self
                .entries
                .get(&id)
                .ok_or(PrefixIndexError::Pool(PagePoolError::CorruptState))?
                .sequence
                .fork_reserved(cached_tokens_u64, reserved_tokens);
            match result {
                Ok(sequence) => break sequence,
                Err(PagePoolError::QuotaExceeded) if self.evict_lru(Some(id))? => {}
                Err(error) => return Err(error.into()),
            }
        };
        Ok(Some(PrefixHit {
            sequence,
            cached_tokens,
        }))
    }

    fn tick(&mut self) -> Result<u64, PrefixIndexError> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(PrefixIndexError::Bookkeeping)?;
        Ok(self.clock)
    }

    fn evict_for(&mut self, tokens: usize, pages: usize) -> Result<(), PrefixIndexError> {
        loop {
            let over_budget = self.entries.len() >= self.limits.entries
                || self
                    .tokens
                    .checked_add(tokens)
                    .is_none_or(|total| total > self.limits.tokens)
                || self
                    .pages
                    .checked_add(pages)
                    .is_none_or(|total| total > self.limits.referenced_pages);
            if !over_budget {
                return Ok(());
            }
            let Some((&id, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.used) else {
                return Err(PrefixIndexError::Bookkeeping);
            };
            self.remove(id)?;
        }
    }

    pub(crate) fn evict_lru(&mut self, excluded: Option<u64>) -> Result<bool, PrefixIndexError> {
        let Some((&id, _)) = self
            .entries
            .iter()
            .filter(|(id, _)| Some(**id) != excluded)
            .min_by_key(|(_, entry)| entry.used)
        else {
            return Ok(false);
        };
        self.remove(id)?;
        Ok(true)
    }

    fn remove(&mut self, id: u64) -> Result<(), PrefixIndexError> {
        let entry = self.entries.get(&id).ok_or(PrefixIndexError::Bookkeeping)?;
        let tokens = self
            .tokens
            .checked_sub(entry.fingerprint.tokens.len())
            .ok_or(PrefixIndexError::Bookkeeping)?;
        let pages = self
            .pages
            .checked_sub(entry.pages)
            .ok_or(PrefixIndexError::Bookkeeping)?;
        let entry = self
            .entries
            .remove(&id)
            .ok_or(PrefixIndexError::Bookkeeping)?;
        self.tokens = tokens;
        self.pages = pages;
        for hash in &entry.fingerprint.chain {
            let empty = self.blocks.get_mut(hash).is_some_and(|ids| {
                ids.retain(|candidate| *candidate != id);
                ids.is_empty()
            });
            if empty {
                self.blocks.remove(hash);
            }
        }
        Ok(())
    }
}
#[derive(Default)]
pub(crate) struct PrefixLedger(Vec<u32>);
impl PrefixLedger {
    pub(crate) fn new(prompt: Vec<u32>) -> Self {
        Self(prompt)
    }
    pub(crate) fn record(&mut self, token: u32, forwarded: bool) {
        if forwarded {
            self.0.push(token);
        }
    }
    pub(crate) fn finish(self, namespace: PrefixNamespace) -> LivePrefix {
        LivePrefix::new(namespace, self.0)
    }

    pub(crate) fn into_tokens(self) -> Vec<u32> {
        self.0
    }
}
pub(crate) fn replace_live(
    entry: &mut Option<LivePrefix>,
    namespace: PrefixNamespace,
    ledger: Option<PrefixLedger>,
) {
    *entry = ledger.map(|ledger| ledger.finish(namespace));
}
impl LivePrefix {
    pub(crate) fn new(namespace: PrefixNamespace, tokens: Vec<u32>) -> Self {
        let mut previous = namespace.0;
        let chain = tokens
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|block| {
                let mut hash = Sha256::new();
                hash.update(BLOCK_DOMAIN);
                hash.update(previous);
                for token in block {
                    hash.update(token.to_le_bytes());
                }
                previous = hash.finalize().into();
                previous
            })
            .collect();
        Self {
            namespace,
            tokens,
            chain,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }
    pub(crate) fn match_len(&self, namespace: PrefixNamespace, prompt: &[u32]) -> usize {
        if namespace != self.namespace || prompt.is_empty() {
            return 0;
        }
        let cap = prompt.len().saturating_sub(1).min(self.tokens.len());
        let candidate_blocks = cap / BLOCK;
        let prompt_chain = Self::new(namespace, prompt[..candidate_blocks * BLOCK].to_vec());
        let equal_blocks = self
            .chain
            .iter()
            .zip(&prompt_chain.chain)
            .take(candidate_blocks)
            .take_while(|(left, right)| left == right)
            .count();
        let candidate = equal_blocks * BLOCK;
        let exact = self.tokens[..candidate]
            .iter()
            .zip(&prompt[..candidate])
            .position(|(left, right)| left != right)
            .unwrap_or(candidate);
        if exact != candidate {
            return exact;
        }
        self.tokens[candidate..cap]
            .iter()
            .zip(&prompt[candidate..cap])
            .position(|(left, right)| left != right)
            .map_or(cap, |offset| candidate + offset)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{DType, PagePool, PagedLayout, PagedPlane};

    fn prefix() -> LivePrefix {
        LivePrefix::new(PrefixNamespace([7; 32]), (0..130).collect())
    }

    fn pool() -> PagePool {
        PagePool::new(
            PagedLayout::new(
                64,
                vec![PagedPlane {
                    dtype: DType::F16,
                    elements_per_token: 8,
                    alignment: 16,
                    label: "cache".to_owned(),
                }],
            )
            .unwrap(),
            8,
            1 << 20,
        )
        .unwrap()
    }

    fn sequence(pool: &PagePool, tokens: usize) -> PagedPrefix {
        let mut sequence = pool.reserve(256).unwrap();
        sequence.commit_write(tokens as u64).unwrap();
        PagedPrefix::new(sequence)
    }
    #[test]
    fn finds_boundaries_partial_blocks_and_prompt_cap() {
        let namespace = PrefixNamespace([7; 32]);
        let mut boundary = (0..130).collect::<Vec<_>>();
        boundary[64] = 999;
        assert_eq!(prefix().match_len(namespace, &boundary), 64);
        let mut middle = (0..130).collect::<Vec<_>>();
        middle[71] = 999;
        assert_eq!(prefix().match_len(namespace, &middle), 71);
        assert_eq!(
            prefix().match_len(namespace, &(0..70).collect::<Vec<_>>()),
            69
        );
        assert_eq!(prefix().match_len(namespace, &[0]), 0);
    }
    #[test]
    fn hash_candidates_are_verified_exactly() {
        let namespace = PrefixNamespace([7; 32]);
        let mut cached = prefix();
        let mut prompt = (0..130).collect::<Vec<_>>();
        prompt[33] = 999;
        cached.chain = LivePrefix::new(namespace, prompt.clone()).chain;
        assert_eq!(cached.match_len(namespace, &prompt), 33);
    }

    #[test]
    fn every_namespace_input_isolated_independently() {
        let base = ["engine", "profile", "weights", "tokenizer", "chat"];
        let expected = PrefixIdentity::new(base).unwrap().namespace_with_nonce(7);
        for index in 0..base.len() {
            let mut changed = base;
            changed[index] = "changed";
            assert_ne!(
                PrefixIdentity::new(changed)
                    .unwrap()
                    .namespace_with_nonce(7),
                expected
            );
        }
        assert_ne!(
            PrefixIdentity::new(base).unwrap().namespace_with_nonce(8),
            expected
        );
    }
    #[test]
    fn ledger_contains_only_completed_inputs() {
        let mut ledger = PrefixLedger::new(vec![1, 2]);
        ledger.record(3, true);
        ledger.record(4, false);
        ledger.record(99, false);
        assert_eq!(ledger.finish(PrefixNamespace([0; 32])).tokens, [1, 2, 3]);
    }
    #[test]
    fn failed_or_cancelled_requests_invalidate_the_live_entry() {
        let namespace = PrefixNamespace([0; 32]);
        let mut entry = None;
        replace_live(&mut entry, namespace, Some(PrefixLedger::new(vec![1])));
        replace_live(&mut entry, namespace, None);
        assert!(entry.is_none());
    }
    #[test]
    fn reused_suffix_reconstructs_the_full_prompt() {
        let namespace = PrefixNamespace([4; 32]);
        let cached = (0..97).collect::<Vec<_>>();
        let mut next = cached.clone();
        next.extend(97..130);
        let prefix = LivePrefix::new(namespace, cached);
        let reused = prefix.match_len(namespace, &next);
        let mut reconstructed = prefix.tokens[..reused].to_vec();
        reconstructed.extend_from_slice(&next[reused..]);
        assert_eq!(reused, 97);
        assert_eq!(reconstructed, next);
    }

    #[test]
    fn unrelated_entry_does_not_evict_an_agent_prefix() {
        let namespace = PrefixNamespace([9; 32]);
        let pool = pool();
        let mut index = PrefixIndex::new(PrefixLimits {
            entries: 3,
            tokens: 768,
            referenced_pages: 12,
        })
        .unwrap();
        let agent = (0..130).collect::<Vec<_>>();
        let unrelated = (1_000..1_100).collect::<Vec<_>>();
        index
            .publish(namespace, agent.clone(), &sequence(&pool, agent.len()))
            .unwrap();
        index
            .publish(
                namespace,
                unrelated.clone(),
                &sequence(&pool, unrelated.len()),
            )
            .unwrap();
        let mut next_turn = agent;
        next_turn.extend(130..140);

        let hit = index.lookup(namespace, &next_turn, 192).unwrap().unwrap();

        assert_eq!(hit.cached_tokens, 130);
        assert_eq!(hit.sequence.logical_len().unwrap(), 130);
        assert_eq!(hit.sequence.reserved_tokens().unwrap(), 192);
    }

    #[test]
    fn least_recent_entry_is_evicted_after_a_hit() {
        let namespace = PrefixNamespace([3; 32]);
        let pool = pool();
        let mut index = PrefixIndex::new(PrefixLimits {
            entries: 2,
            tokens: 512,
            referenced_pages: 12,
        })
        .unwrap();
        for start in [0, 1_000] {
            let tokens = (start..start + 100).collect::<Vec<_>>();
            index
                .publish(namespace, tokens, &sequence(&pool, 100))
                .unwrap();
        }
        let first = (0..110).collect::<Vec<_>>();
        assert!(index.lookup(namespace, &first, 128).unwrap().is_some());
        let third = (2_000..2_100).collect::<Vec<_>>();
        index
            .publish(namespace, third.clone(), &sequence(&pool, 100))
            .unwrap();

        assert!(index.lookup(namespace, &first, 128).unwrap().is_some());
        assert!(
            index
                .lookup(namespace, &(1_000..1_110).collect::<Vec<_>>(), 128)
                .unwrap()
                .is_none()
        );
        assert!(index.lookup(namespace, &third, 128).unwrap().is_some());
    }

    #[test]
    fn bookkeeping_drift_is_reported_without_removing_entries() {
        let namespace = PrefixNamespace([5; 32]);
        let pool = pool();
        let mut index = PrefixIndex::new(PrefixLimits {
            entries: 2,
            tokens: 256,
            referenced_pages: 8,
        })
        .unwrap();
        let tokens = (0..100).collect::<Vec<_>>();
        let sequence = sequence(&pool, tokens.len());
        index.publish(namespace, tokens.clone(), &sequence).unwrap();
        index.tokens = 0;

        assert!(matches!(
            index.remove(0),
            Err(PrefixIndexError::Bookkeeping)
        ));
        assert!(index.entries.contains_key(&0));

        index.entries.clear();
        index.tokens = usize::MAX;
        assert!(matches!(
            index.evict_for(1, 1),
            Err(PrefixIndexError::Bookkeeping)
        ));
    }
}
