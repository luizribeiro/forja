use forja_host::{SamplingParams, speculative::VerifyCut};

/// Deterministic proposals for exact target-stream tests.
#[derive(Clone, Copy, Debug)]
pub enum TestDrafter {
    /// Always follows its own proposed prefix.
    Oracle,
    /// Always changes the target selection.
    Adversary,
    /// Matches only the requested leading rows.
    OraclePrefix(usize),
    /// Changes the first row, then follows the resulting prefix.
    WrongFirstRightSecond,
    /// Continues a seeded pseudo-random sequence.
    Random(u64),
}

impl TestDrafter {
    /// Proposes tokens against the target evaluated on each proposed prefix.
    ///
    /// # Panics
    /// Panics on an invalid fixture or a failed CPU operation.
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
/// Observable output and paths from one fresh request.
#[derive(Default, Debug)]
pub struct Trace {
    /// Selected output stream.
    pub tokens: Vec<u32>,
    /// Accepted draft counts in verify order.
    pub accepted: Vec<u32>,
    /// Verify row counts in execution order.
    pub shapes: Vec<u32>,
    /// Ordinary decode steps.
    pub plain: usize,
    /// Terminal token boundary, if reached.
    pub cut: Option<VerifyCut>,
}

/// Compares fresh plain and draft traces across supported sampling modes.
///
/// # Panics
/// Panics when paths, oracle acceptance or output streams differ.
pub fn assert_draft_exact(
    verify_expected: bool,
    mut run: impl FnMut(SamplingParams, Option<TestDrafter>) -> Trace,
) {
    for sampling in [
        SamplingParams::default(),
        SamplingParams {
            temperature: 0.7,
            seed: 0xfeed_beef_dead_cafe,
            ..SamplingParams::default()
        },
        SamplingParams {
            temperature: 0.7,
            top_k: 7,
            top_p: 0.9,
            seed: 33,
        },
        SamplingParams {
            temperature: 0.7,
            top_p: 0.8,
            seed: 7,
            ..SamplingParams::default()
        },
    ] {
        let plain = run(sampling, None);
        for drafter in [
            TestDrafter::Oracle,
            TestDrafter::Adversary,
            TestDrafter::OraclePrefix(1),
            TestDrafter::OraclePrefix(2),
            TestDrafter::WrongFirstRightSecond,
            TestDrafter::Random(0),
            TestDrafter::Random(7),
        ] {
            let trace = run(sampling, Some(drafter));
            assert_eq!(
                trace.tokens, plain.tokens,
                "draft token stream differs: sampling={sampling:?}, drafter={drafter:?}"
            );
            assert_eq!(trace.shapes.len(), trace.accepted.len());
            assert_eq!(
                !trace.shapes.is_empty(),
                verify_expected,
                "verify path differs: sampling={sampling:?}, drafter={drafter:?}"
            );
            if matches!(drafter, TestDrafter::Oracle) {
                let mut remaining = trace.tokens.len();
                for (round, (shape, accepted)) in
                    trace.shapes.iter().zip(&trace.accepted).enumerate()
                {
                    let expected = (shape - 1).min(u32::try_from(remaining - 1).unwrap());
                    let boundary_cut = round + 1 == trace.shapes.len()
                        && matches!(
                            trace.cut,
                            Some(VerifyCut::Stop | VerifyCut::Eos | VerifyCut::Activate)
                        );
                    if boundary_cut {
                        assert!(
                            *accepted <= expected,
                            "a cut cannot extend the oracle prefix"
                        );
                    } else {
                        assert_eq!(
                            *accepted, expected,
                            "oracle must accept every fitting draft"
                        );
                    }
                    remaining -= usize::try_from(accepted + 1).unwrap();
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "draft token stream differs")]
fn exactness_helper_refuses_a_changed_token_stream() {
    assert_draft_exact(false, |_, drafter| Trace {
        tokens: vec![u32::from(drafter.is_some())],
        ..Trace::default()
    });
}

#[test]
#[should_panic(expected = "verify path differs")]
fn exactness_helper_refuses_silent_plain_fallback() {
    assert_draft_exact(true, |_, _| Trace {
        tokens: vec![7; 3],
        ..Trace::default()
    });
}

#[test]
#[should_panic(expected = "oracle must accept every fitting draft")]
fn exactness_helper_refuses_incomplete_oracle_acceptance() {
    assert_draft_exact(true, |_, _| Trace {
        tokens: vec![7; 3],
        shapes: vec![3],
        accepted: vec![0],
        ..Trace::default()
    });
}

#[test]
fn exactness_helper_allows_a_terminal_boundary_to_shorten_the_oracle_prefix() {
    for cut in [VerifyCut::Stop, VerifyCut::Eos, VerifyCut::Activate] {
        assert_draft_exact(true, |_, _| Trace {
            tokens: vec![7, 33],
            accepted: vec![0],
            shapes: vec![3],
            cut: Some(cut),
            ..Trace::default()
        });
    }
}
