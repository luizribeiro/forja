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
/// Final serving observables captured before a lane releases its storage.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct RequestOutput {
    /// Concatenated content events from the real stream sink.
    pub streamed_text: String,
    /// Prompt, cached prompt and completion token usage, in that order.
    pub usage: [u32; 3],
    /// Number of tokens in the accepted prefix ledger.
    pub ledger_len: u64,
    /// Logical length of the lane's KV storage.
    pub kv_logical_len: u64,
}

/// Observable output and paths from one fresh request.
#[derive(Default, Debug)]
pub struct Trace {
    /// Serving observables, when this trace drives a serving lane.
    pub output: Option<RequestOutput>,
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

/// Path checks for a real source or the deterministic test-drafter suite.
pub struct ExactnessChecks {
    verify_expected: bool,
    oracle_acceptance: bool,
    sampling: Option<SamplingParams>,
}

impl ExactnessChecks {
    /// Restricts a fixture to one sampling configuration while preserving all drafter checks.
    #[must_use]
    pub const fn for_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = Some(sampling);
        self
    }

    /// Checks a real host source whose proposals can legitimately be rejected.
    #[must_use]
    pub const fn source(verify_expected: bool) -> Self {
        Self {
            verify_expected,
            oracle_acceptance: false,
            sampling: None,
        }
    }
}

impl From<bool> for ExactnessChecks {
    fn from(verify_expected: bool) -> Self {
        Self {
            verify_expected,
            oracle_acceptance: true,
            sampling: None,
        }
    }
}

/// Sampling configurations shared by exact-stream fixtures.
#[must_use]
pub fn sampling_modes() -> [SamplingParams; 4] {
    [
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
    ]
}

/// Compares fresh plain and draft traces across supported sampling modes.
///
/// # Panics
/// Panics when paths, oracle acceptance or output streams differ.
pub fn assert_draft_exact(
    checks: impl Into<ExactnessChecks>,
    mut run: impl FnMut(SamplingParams, Option<TestDrafter>) -> Trace,
) {
    let checks = checks.into();
    let samplings = checks
        .sampling
        .map_or_else(|| sampling_modes().to_vec(), |sampling| vec![sampling]);
    for sampling in samplings {
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
            assert_eq!(
                trace.output, plain.output,
                "draft request output differs: sampling={sampling:?}, drafter={drafter:?}"
            );
            assert_eq!(trace.shapes.len(), trace.accepted.len());
            assert_eq!(
                !trace.shapes.is_empty(),
                checks.verify_expected,
                "verify path differs: sampling={sampling:?}, drafter={drafter:?}"
            );
            if checks.oracle_acceptance && matches!(drafter, TestDrafter::Oracle) {
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
fn exactness_helper_accepts_rejected_real_source_proposals() {
    assert_draft_exact(ExactnessChecks::source(true), |_, source| Trace {
        tokens: vec![7; 7],
        shapes: if source.is_some() { vec![3; 3] } else { vec![] },
        accepted: if source.is_some() {
            vec![0, 1, 2]
        } else {
            vec![]
        },
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

#[test]
fn exactness_helper_refuses_each_changed_serving_observable() {
    for field in 0..6 {
        let result = std::panic::catch_unwind(|| {
            assert_draft_exact(false, |_, drafter| {
                let mut output = RequestOutput::default();
                if drafter.is_some() {
                    match field {
                        0 => output.streamed_text.push('x'),
                        1..=3 => output.usage[field - 1] = 1,
                        4 => output.ledger_len = 1,
                        _ => output.kv_logical_len = 1,
                    }
                }
                Trace {
                    output: Some(output),
                    tokens: vec![7],
                    ..Trace::default()
                }
            });
        });
        assert!(result.is_err(), "observable {field} was not compared");
    }
}

#[test]
fn sampling_override_runs_every_drafter_with_the_requested_configuration() {
    let mode = SamplingParams {
        temperature: 0.5,
        seed: 9,
        ..SamplingParams::default()
    };
    let mut calls = 0;
    assert_draft_exact(
        ExactnessChecks::from(false).for_sampling(mode),
        |sampling, _| {
            assert_eq!(sampling, mode);
            calls += 1;
            Trace {
                tokens: vec![7],
                ..Trace::default()
            }
        },
    );
    assert_eq!(calls, 8);
}
