use super::*;

fn check_cut(rows: u32, accepted: u32) {
    let mut fixture = Fixture::new(rows, false);
    assert!(
        fixture
            .publication
            .publish_kv(&mut fixture.host, rows)
            .is_err()
    );
    fixture
        .publication
        .publish_kv(&mut fixture.host, accepted)
        .unwrap();
    assert_eq!(fixture.logical_len(), u64::from(accepted) + 1);
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
cut!(kv_rejects_the_only_draft, 2, 0);
cut!(kv_accepts_the_only_draft, 2, 1);
cut!(kv_rejects_both_drafts, 3, 0);
cut!(kv_keeps_the_first_draft, 3, 1);
cut!(kv_keeps_both_drafts, 3, 2);
