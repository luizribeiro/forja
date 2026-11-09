#[path = "speculative/cpu.rs"]
mod cpu;
#[path = "speculative/request.rs"]
mod request;
use crate::common::draft::{TestDrafter, assert_draft_exact};
use forja_host::speculative::VerifyCut;
use request::Request;

#[test]
fn draft_requests_are_exact_at_odd_positions_and_output_budgets() {
    for rule_cap in [1, 2] {
        let mut accepted = std::collections::BTreeSet::new();
        for start in [0_u32, 7, 33, 4097] {
            for max_tokens in [1, 2, 7, 33] {
                let request = Request {
                    prefix: (0..=start).map(|i| i % 33).collect(),
                    max_tokens,
                    rule_cap,
                    ..Request::default()
                };
                assert_draft_exact(max_tokens > 1, |sampling, drafter| {
                    let trace = request.run(sampling, drafter);
                    assert_eq!(u64::try_from(trace.tokens.len()).unwrap(), max_tokens);
                    assert_eq!(trace.cut, Some(VerifyCut::MaxTokens));
                    accepted.extend(trace.accepted.iter().copied());
                    trace
                });
            }
        }
        assert_eq!(accepted, (0..=rule_cap).collect());
    }
}

#[test]
fn draft_requests_fall_back_or_shrink_to_available_round_headroom() {
    for (count, cap, rows, context, reserved, expected) in [
        (0, 2, vec![2, 3], 8192, 8192, None),
        (1, 2, vec![2, 3], 8192, 8192, Some(2)),
        (1, 2, vec![3], 8192, 8192, None),
        (2, 0, vec![2, 3], 8192, 8192, None),
        (2, 2, vec![], 8192, 8192, None),
        (2, 2, vec![2, 3], 4, 8192, Some(2)),
        (2, 2, vec![2, 3], 8192, 4, Some(2)),
        (2, 2, vec![2, 3], 3, 8192, None),
        (2, 2, vec![2, 3], 8192, 2, None),
    ] {
        let mut request = Request {
            draft_count: count,
            rule_cap: cap,
            max_context: context,
            reserved,
            ..Request::default()
        };
        request.shapes.retain(|shape| rows.contains(&shape.tokens));
        assert_draft_exact(expected.is_some(), |sampling, drafter| {
            let trace = request.run(sampling, drafter);
            assert_eq!(trace.shapes.first().copied(), drafter.and(expected));
            trace
        });
    }
}

#[test]
fn draft_requests_stop_at_the_same_accepted_token_boundary() {
    for reason in [
        VerifyCut::Stop,
        VerifyCut::Eos,
        VerifyCut::Activate,
        VerifyCut::MaxTokens,
    ] {
        for index in [0, 1, 2, 3, 6] {
            let request = Request {
                max_tokens: 7,
                cut: Some((index, reason)),
                ..Request::default()
            };
            assert_draft_exact(true, |sampling, drafter| {
                let trace = request.run(sampling, drafter);
                assert_eq!(trace.tokens.len(), index + 1);
                assert_eq!(trace.cut, Some(reason));
                trace
            });
        }
    }
}

#[test]
fn mixed_drafts_accept_only_the_contiguous_prefix_before_cuts() {
    for rule_cap in [1, 2] {
        let request = Request {
            rule_cap,
            max_tokens: 7,
            ..Request::default()
        };
        for (drafter, accepted) in [
            (TestDrafter::OraclePrefix(1), 1),
            (TestDrafter::WrongFirstRightSecond, 0),
        ] {
            let trace = request.run(forja_host::SamplingParams::default(), Some(drafter));
            assert_eq!(trace.shapes[0], rule_cap + 1);
            assert!(trace.accepted.iter().all(|count| *count == accepted));
        }
    }
    for reason in [VerifyCut::Stop, VerifyCut::Eos, VerifyCut::MaxTokens] {
        let request = Request {
            cut: Some((0, reason)),
            ..Request::default()
        };
        let trace = request.run(
            forja_host::SamplingParams::default(),
            Some(TestDrafter::OraclePrefix(1)),
        );
        assert_eq!(trace.shapes, [3]);
        assert_eq!(trace.accepted, [0]);
        assert_eq!(trace.tokens.len(), 1);
        assert_eq!(trace.cut, Some(reason));
    }
    let request = Request {
        max_tokens: 2,
        ..Request::default()
    };
    let trace = request.run(
        forja_host::SamplingParams::default(),
        Some(TestDrafter::OraclePrefix(2)),
    );
    assert_eq!(trace.shapes, [2]);
    assert_eq!(trace.accepted, [1]);
    assert_eq!(trace.cut, Some(VerifyCut::MaxTokens));
}
