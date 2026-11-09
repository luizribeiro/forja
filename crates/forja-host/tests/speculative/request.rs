use super::cpu::Target;
use crate::common::draft::TestDrafter;
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, SamplingParams,
    speculative::{VerifyAcceptance, VerifyBudget, VerifyCut, select_verify_shape},
};

pub(super) struct Request {
    pub prefix: Vec<u32>,
    pub max_tokens: u64,
    pub max_context: u64,
    pub reserved: u64,
    pub rule_cap: u32,
    pub draft_count: usize,
    pub shapes: Vec<BatchShape>,
    pub cut: Option<(usize, VerifyCut)>,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            prefix: vec![1, 7, 32],
            max_tokens: 33,
            max_context: 8192,
            reserved: 8192,
            rule_cap: 2,
            draft_count: 2,
            shapes: (2..=3)
                .map(|tokens| BatchShape {
                    phase: BatchPhase::Verify,
                    bucket: 1,
                    tokens,
                    selection: BatchSelection::Combined,
                })
                .collect(),
            cut: None,
        }
    }
}

#[derive(Default, Debug)]
pub(super) struct Trace {
    pub tokens: Vec<u32>,
    pub accepted: Vec<u32>,
    pub shapes: Vec<u32>,
    pub plain: usize,
    pub cut: Option<VerifyCut>,
}

impl Request {
    pub(super) fn run(&self, sampling: SamplingParams, mut drafter: Option<TestDrafter>) -> Trace {
        let target = Target::new(33, sampling);
        let mut history = self.prefix.clone();
        let mut trace = Trace::default();
        loop {
            let start = u64::try_from(history.len().checked_sub(1).unwrap()).unwrap();
            let left = self
                .max_tokens
                .checked_sub(u64::try_from(trace.tokens.len()).unwrap())
                .unwrap();
            let Some(budget) =
                VerifyBudget::new(start, self.rule_cap, left, self.max_context, self.reserved)
            else {
                break;
            };
            let drafts = drafter.as_mut().map_or_else(Vec::new, |drafter| {
                drafter.propose(&history, self.draft_count, target.vocab, |prefix| {
                    target.sample(prefix)
                })
            });
            let (tokens, cut) =
                if let Some(fed) = select_verify_shape(&self.shapes, &drafts, budget) {
                    trace.shapes.push(fed.shape().tokens);
                    let mut verified = history.clone();
                    let mut samples = vec![target.sample(&verified)];
                    for token in fed.tokens() {
                        verified.push(*token);
                        samples.push(target.sample(&verified));
                    }
                    let mut scanned = trace.tokens.len();
                    let decision = VerifyAcceptance::new(fed, &samples, target.vocab)
                        .unwrap()
                        .scan(|_| {
                            let cut = self.cut_at(scanned);
                            scanned += 1;
                            Ok::<_, ()>(cut)
                        })
                        .unwrap();
                    trace.accepted.push(decision.accepted());
                    assert_eq!(decision.pending(), *decision.tokens().last().unwrap());
                    (decision.tokens().to_vec(), decision.cut())
                } else {
                    trace.plain += 1;
                    (
                        vec![target.sample(&history)],
                        self.cut_at(trace.tokens.len())
                            .or((left == 1).then_some(VerifyCut::MaxTokens)),
                    )
                };
            history.extend_from_slice(&tokens);
            trace.tokens.extend(tokens);
            trace.cut = cut;
            if cut.is_some() {
                break;
            }
        }
        trace
    }

    fn cut_at(&self, index: usize) -> Option<VerifyCut> {
        self.cut.filter(|(at, _)| *at == index).map(|(_, cut)| cut)
    }
}

#[test]
fn request_accepts_oracle_rounds_and_discards_adversarial_successors() {
    let request = Request {
        max_tokens: 7,
        ..Request::default()
    };
    let oracle = request.run(SamplingParams::default(), Some(TestDrafter::Oracle));
    assert_eq!(oracle.tokens.len(), 7);
    assert_eq!(oracle.accepted, [2, 2]);
    assert_eq!(oracle.shapes, [3, 3]);
    assert_eq!(oracle.plain, 1);
    assert_eq!(oracle.cut, Some(VerifyCut::MaxTokens));
    let adversary = request.run(SamplingParams::default(), Some(TestDrafter::Adversary));
    let plain = request.run(SamplingParams::default(), None);
    assert_eq!(adversary.tokens, plain.tokens);
    assert_eq!(adversary.accepted, [0; 6]);
    assert_eq!(adversary.plain, 1);
}
