use super::*;
use crate::{CommandList, DType, PagedCopyDirection, cost::tests::tensor};

#[test]
fn strided_copy_counts_elements_instead_of_backing_storage() {
    let allocation = tensor(1, DType::F32, &[33, 7]);
    let input = Tensor::from_allocation(
        allocation.buffer(),
        allocation.layout().permute(&[1, 0]).unwrap(),
        false,
    )
    .unwrap();
    let output = tensor(2, DType::F16, &[7, 33]);
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&input], &output).unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(
        (cost.read_bytes, cost.write_bytes),
        (7 * 33 * 4, 7 * 33 * 2)
    );
    assert_eq!(cost.flops, 0);
}

#[test]
fn paged_copy_counts_transferred_rows_in_both_directions() {
    let ordinary = tensor(1, DType::F32, &[7, 8, 128]);
    let pool = tensor(2, DType::F16, &[99, 16, 8 * 128]);
    let table = tensor(3, DType::U32, &[3, 2]);
    let generations = tensor(4, DType::U32, &[99]);
    for direction in [
        PagedCopyDirection::IntoPages,
        PagedCopyDirection::OutOfPages,
    ] {
        let (input, output, expected) = match direction {
            PagedCopyDirection::IntoPages => (
                &ordinary,
                &pool,
                (7 * 8 * 128 * 4 + 3 * 2 * 4 + 99 * 4, 7 * 8 * 128 * 2),
            ),
            PagedCopyDirection::OutOfPages => (
                &pool,
                &ordinary,
                (7 * 8 * 128 * 2 + 3 * 2 * 4 + 99 * 4, 7 * 8 * 128 * 4),
            ),
        };
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::PagedCopy {
                    direction,
                    start: 26,
                    page_tokens: 16,
                },
                &[input, &table, &generations],
                output,
            )
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        assert_eq!((cost.read_bytes, cost.write_bytes), expected);
        assert_eq!(cost.flops, 0);
    }
}
