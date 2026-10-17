use sha2::{Digest, Sha256};
const BLOCK: usize = 64;
const BLOCK_DOMAIN: &[u8] = b"forja-prefix-block-v1";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PrefixNamespace(pub(crate) [u8; 32]);
#[derive(Debug)]
pub(crate) struct LivePrefix {
    namespace: PrefixNamespace,
    tokens: Vec<u32>,
    chain: Vec<[u8; 32]>,
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
    fn prefix() -> LivePrefix {
        LivePrefix::new(PrefixNamespace([7; 32]), (0..130).collect())
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
}
