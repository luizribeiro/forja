use super::*;

fn check_cut(rows: u32, accepted: u32) {
    let mut fixture = Fixture::new(rows, true);
    assert!(
        fixture
            .publication
            .publish(&mut fixture.host, rows)
            .is_err()
    );
    assert!(
        fixture
            .publication
            .publish(&mut fixture.host, accepted)
            .is_err()
    );
    fixture.finish();
    fixture
        .publication
        .publish(&mut fixture.host, accepted)
        .unwrap();
    assert_eq!(fixture.logical_len(), u64::from(accepted) + 1);
    let position = fixture
        .host
        .state_lane_checkpoint(fixture.pool.rep(), 1)
        .unwrap()
        .unwrap()
        .position();
    assert_eq!(
        (
            position.tokens,
            position.sampler,
            position.grammar,
            position.parser
        ),
        (
            u64::from(accepted) + 1,
            u64::from(accepted),
            u64::from(accepted) + 1,
            u64::from(accepted) + 1
        )
    );
    assert!(
        fixture
            .publication
            .publish(&mut fixture.host, accepted)
            .is_err()
    );
    fixture.rollback();
}
macro_rules! cut {
    ($name:ident, $rows:expr, $accepted:expr) => {
        #[test]
        fn $name() {
            check_cut($rows, $accepted);
        }
    };
}
cut!(carry_rejects_the_only_draft, 2, 0);
cut!(carry_accepts_the_only_draft, 2, 1);
cut!(carry_rejects_both_drafts, 3, 0);
cut!(carry_keeps_the_first_draft, 3, 1);
cut!(carry_keeps_both_drafts, 3, 2);

#[test]
fn rejected_protected_rows_rebind_the_device_page_table() {
    use std::sync::atomic::Ordering;
    let mut fixture = Fixture::new_at(3, true, 3);
    fixture.finish();
    let binding = fixture
        .host
        .table
        .get(&fixture.slot)
        .unwrap()
        .binding
        .clone();
    for dirty in &binding.lane(1).unwrap().table_dirty {
        dirty.store(false, Ordering::Release);
    }
    fixture.publication.publish(&mut fixture.host, 0).unwrap();
    assert_eq!(fixture.logical_len(), 4);
    assert!(
        binding
            .lane(1)
            .unwrap()
            .table_dirty
            .iter()
            .all(|dirty| dirty.load(Ordering::Acquire))
    );
    let (checkpoint, _) = fixture
        .host
        .begin_paged_lane_update(fixture.pool.rep(), 1, 4, 1, Some(StateAction::Step))
        .unwrap();
    let slot = fixture.host.table.get(&fixture.slot).unwrap();
    let table = slot.tables[slot.active_bank][1];
    let actual = fixture
        .host
        .backend
        .read(
            &fixture
                .host
                .entry(&Resource::new_borrow(table))
                .unwrap()
                .tensor,
        )
        .unwrap();
    let prefix = fixture
        .host
        .paged_lane_prefix(fixture.pool.rep(), 1)
        .unwrap();
    let sequence = prefix.sequence.lock().unwrap();
    let table = sequence.table().unwrap();
    let expected: Vec<_> = table
        .entries()
        .iter()
        .flat_map(|entry| [entry.page().to_ne_bytes(), entry.generation().to_ne_bytes()].concat())
        .collect();
    assert_eq!(&actual[..expected.len()], expected);
    drop(sequence);
    fixture
        .host
        .rollback_paged_lane_append(fixture.pool.rep(), 1, checkpoint.unwrap())
        .unwrap();
}
