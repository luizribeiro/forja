use super::*;

fn prompt_with_distinct_initial_tokens(sampling: SamplingParams) -> (Vec<u32>, Vec<u32>) {
    for length in 7..=15 {
        for token in 0..4 {
            let prompt = vec![token; length];
            let mut history = prompt.clone();
            let tokens: Vec<_> = (0..5)
                .map(|_| {
                    let token = target(&history, sampling);
                    history.push(token);
                    token
                })
                .collect();
            if (0..4).all(|index| !tokens[..index].contains(&tokens[index])) {
                return (prompt, tokens);
            }
        }
    }
    panic!("no distinct-token prompt for {sampling:?}");
}

#[derive(Clone, Copy)]
enum CutCase {
    EosFirstDraft,
    EosSecondDraft,
    EosBonus,
    StopAcrossPrefill,
    MidTextStop,
    StopAtLimit,
    LaterRoundFirstRow,
}

#[test]
fn serving_cuts_match_all_plain_observables_for_every_test_drafter() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    for scenario in [
        CutCase::EosFirstDraft,
        CutCase::EosSecondDraft,
        CutCase::EosBonus,
        CutCase::StopAcrossPrefill,
        CutCase::MidTextStop,
        CutCase::StopAtLimit,
        CutCase::LaterRoundFirstRow,
    ] {
        assert_draft_exact(true, |sampling, drafter| {
            let (prompt, tokens) = prompt_with_distinct_initial_tokens(sampling);
            let (stop_token, range, max_tokens) = match scenario {
                CutCase::EosFirstDraft => (Some(tokens[1]), None, 9),
                CutCase::EosSecondDraft => (Some(tokens[2]), None, 9),
                CutCase::EosBonus => (Some(tokens[3]), None, 9),
                CutCase::StopAcrossPrefill => (None, Some(0..2), 9),
                CutCase::MidTextStop => (None, Some(1..4), 9),
                CutCase::StopAtLimit => (None, Some(1..4), 4),
                CutCase::LaterRoundFirstRow => (None, Some(1..5), 9),
            };
            let stop_text = range.map(|range| {
                tokens[range]
                    .iter()
                    .map(|id| char::from(b'a' + u8::try_from(*id).unwrap()))
                    .collect()
            });
            let trace = runtime
                .block_on(direct::run_case(
                    sampling,
                    drafter,
                    2,
                    2,
                    direct::Case {
                        prompt,
                        max_tokens,
                        stop_token,
                        stop_text,
                    },
                ))
                .unwrap();
            assert!(trace.output.is_some());
            if matches!(drafter, Some(TestDrafter::Oracle)) {
                if matches!(scenario, CutCase::EosBonus) {
                    assert_eq!(trace.accepted, [2]);
                }
                if matches!(scenario, CutCase::LaterRoundFirstRow) {
                    assert_eq!(trace.accepted, [2, 0]);
                }
            }
            if matches!(scenario, CutCase::MidTextStop | CutCase::StopAtLimit) {
                assert_ne!(trace.output.as_ref().unwrap().streamed_text, "");
                assert_eq!(trace.cut, Some(VerifyCut::Stop));
            }
            trace
        });
    }
}
