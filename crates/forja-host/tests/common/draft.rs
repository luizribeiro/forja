#[derive(Clone, Copy, Debug)]
pub enum TestDrafter {
    Oracle,
    Adversary,
    OraclePrefix(usize),
    WrongFirstRightSecond,
    Random(u64),
}

impl TestDrafter {
    pub fn propose(
        &mut self,
        prefix: &[u32],
        count: usize,
        vocab: u32,
        mut target: impl FnMut(&[u32]) -> u32,
    ) -> Vec<u32> {
        assert!(vocab >= 2);
        let mut history = prefix.to_vec();
        for row in 0..count {
            let token = match self {
                Self::Oracle
                | Self::Adversary
                | Self::OraclePrefix(_)
                | Self::WrongFirstRightSecond => {
                    let token = target(&history);
                    let correct = match self {
                        Self::Oracle => true,
                        Self::OraclePrefix(count) => row < *count,
                        Self::WrongFirstRightSecond => row > 0,
                        _ => false,
                    };
                    if correct { token } else { (token + 1) % vocab }
                }
                Self::Random(seed) => {
                    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    u32::try_from(*seed >> 32).unwrap() % vocab
                }
            };
            assert!(token < vocab);
            history.push(token);
        }
        history[prefix.len()..].to_vec()
    }
}

#[test]
fn oracle_and_adversary_consult_their_own_proposed_prefixes() {
    let prefix = [1, 7, 33];
    for mut drafter in [TestDrafter::Oracle, TestDrafter::Adversary] {
        let mut seen = Vec::new();
        let drafts = drafter.propose(&prefix, 7, 33, |history| {
            seen.push(history.to_vec());
            history.iter().sum::<u32>() % 33
        });
        let mut history = prefix.to_vec();
        for (row, draft) in drafts.into_iter().enumerate() {
            assert_eq!(seen[row], history);
            assert_eq!(
                draft == history.iter().sum::<u32>() % 33,
                matches!(drafter, TestDrafter::Oracle)
            );
            history.push(draft);
        }
        assert_eq!(
            drafter.propose(&prefix, 0, 33, |_| unreachable!()),
            [] as [u32; 0]
        );
    }
}

#[test]
fn mixed_drafts_follow_the_requested_match_pattern() {
    for (mut drafter, matches) in [
        (TestDrafter::OraclePrefix(0), [false, false, false]),
        (TestDrafter::OraclePrefix(1), [true, false, false]),
        (TestDrafter::OraclePrefix(2), [true, true, false]),
        (TestDrafter::WrongFirstRightSecond, [false, true, true]),
    ] {
        let mut seen = Vec::new();
        let drafts = drafter.propose(&[7], 3, 33, |prefix| {
            seen.push(prefix.to_vec());
            prefix.iter().sum::<u32>() % 33
        });
        let mut prefix = vec![7];
        for (row, token) in drafts.into_iter().enumerate() {
            assert_eq!(seen[row], prefix);
            assert_eq!(token == prefix.iter().sum::<u32>() % 33, matches[row]);
            prefix.push(token);
        }
    }
}

#[test]
fn random_drafts_are_seeded_in_range_and_continue_across_rounds() {
    for vocab in [2, 7, 33, 4097, u32::MAX] {
        let generate =
            |seed, count| TestDrafter::Random(seed).propose(&[], count, vocab, |_| unreachable!());
        let expected = generate(0, 33);
        assert_eq!(generate(0, 33), expected);
        assert_ne!(generate(7, 33), expected);
        assert!(expected.iter().all(|token| *token < vocab));
        let mut drafter = TestDrafter::Random(0);
        let mut split = drafter.propose(&[], 7, vocab, |_| unreachable!());
        split.extend(drafter.propose(&[], 26, vocab, |_| unreachable!()));
        assert_eq!(split, expected);
    }
}
