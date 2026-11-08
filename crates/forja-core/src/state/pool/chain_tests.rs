use super::*;
use crate::{DType, StatePlane};

fn pool() -> StatePool {
    let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![7, 33]).unwrap()]).unwrap();
    StatePool::new(layout, 1, 65536).unwrap()
}

#[test]
fn chain_publication_retains_the_source_and_final_output() {
    let pool = pool();
    let mut sequence = pool.reserve().unwrap();
    let source = sequence.checkpoint().unwrap();
    let chain = sequence.begin_chain(1, 2).unwrap();
    let latest = sequence.checkpoint_latest().unwrap();
    assert_eq!(source.version(), chain.input_version());
    assert_eq!(latest.version(), 2);
    assert_eq!(sequence.predecessor(&chain).unwrap().version(), 0);
    assert_eq!(sequence.rewind(&latest), Err(PagePoolError::BankInFlight));
    sequence.mark_submitted(&chain).unwrap();
    assert_eq!(
        sequence.finish(&chain, true),
        Err(PagePoolError::BankInFlight)
    );
    pool.observe_completion(1).unwrap();
    sequence.finish(&chain, true).unwrap();
    sequence.rewind(&latest).unwrap();
    let next = sequence.begin(2).unwrap();
    assert_eq!(next.input_version(), 2);
    assert_eq!(next.output_versions().collect::<Vec<_>>(), &[1]);
    sequence.cancel_unsubmitted(&next).unwrap();
    sequence.rewind(&source).unwrap();
    assert_eq!(sequence.begin(3).unwrap().input_version(), 0);
}

#[test]
fn pinned_versions_refuse_the_whole_chain_without_losing_plain_capacity() {
    let pool = pool();
    let mut sequence = pool.reserve().unwrap();
    let source = sequence.checkpoint().unwrap();
    let first = sequence.begin(1).unwrap();
    sequence.mark_submitted(&first).unwrap();
    pool.observe_completion(1).unwrap();
    sequence.finish(&first, true).unwrap();
    assert_eq!(
        sequence.begin_chain(2, 2).err(),
        Some(PagePoolError::BankInFlight)
    );
    let plain = sequence.begin(2).unwrap();
    assert_eq!(plain.output_versions().collect::<Vec<_>>(), &[2]);
    sequence.cancel_unsubmitted(&plain).unwrap();
    drop(source);
    let chain = sequence.begin_chain(3, 2).unwrap();
    assert_eq!(chain.input_version(), 1);
    assert_eq!(chain.output_versions().collect::<Vec<_>>(), &[0, 2]);
}

#[test]
fn cancelled_or_failed_chains_never_publish_their_checkpoints() {
    let pool = pool();
    for submitted in [false, true] {
        let mut sequence = pool.reserve().unwrap();
        let source = sequence.checkpoint().unwrap();
        let chain = sequence.begin_chain(1, 2).unwrap();
        let latest = sequence.checkpoint_latest().unwrap();
        if submitted {
            sequence.mark_submitted(&chain).unwrap();
            assert_eq!(
                sequence.finish(&chain, false),
                Err(PagePoolError::BankInFlight)
            );
            pool.observe_completion(1).unwrap();
            sequence.finish(&chain, false).unwrap();
        } else {
            sequence.cancel_unsubmitted(&chain).unwrap();
        }
        assert_eq!(
            sequence.rewind(&latest),
            Err(PagePoolError::InvalidCheckpoint)
        );
        drop(latest);
        sequence.rewind(&source).unwrap();
        let retry = sequence.begin_chain(2, 2).unwrap();
        assert_eq!(retry.input_version(), 0);
        assert_eq!(retry.output_versions().collect::<Vec<_>>(), &[1, 2]);
    }
}
