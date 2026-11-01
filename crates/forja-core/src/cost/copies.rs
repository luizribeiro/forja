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
            PagedCopyDirection::IntoPages => {
                (&ordinary, &pool, (7 * 8 * 128 * 4 + 24, 7 * 8 * 128 * 2))
            }
            PagedCopyDirection::OutOfPages => {
                (&pool, &ordinary, (7 * 8 * 128 * 2 + 24, 7 * 8 * 128 * 4))
            }
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

#[test]
fn private_plane_copies_count_full_state_without_arithmetic() {
    for shape in [vec![32, 128, 128], vec![3, 7, 33]] {
        let source = tensor(1, DType::F32, &shape);
        let destination = tensor(2, DType::F32, &shape);
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&source], &destination)
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        let bytes = source.layout().element_count() * 4;
        assert_eq!((cost.read_bytes, cost.write_bytes), (bytes, bytes));
        assert_eq!((cost.flops, cost.scalar_ops, cost.sfu_ops), (0, 0, 0));
    }
}

#[test]
fn partial_page_cow_copy_counts_only_the_shared_tail() {
    let pool = tensor(1, DType::F16, &[3, 64, 2048]);
    for tokens in [1, 7, 33] {
        let view = |page| {
            let layout = pool
                .layout()
                .slice(&[
                    crate::Slice::new(page, 1, 1).unwrap(),
                    crate::Slice::new(0, tokens, 1).unwrap(),
                    crate::Slice::new(0, 2048, 1).unwrap(),
                ])
                .unwrap();
            Tensor::from_allocation(pool.buffer(), layout, true).unwrap()
        };
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&view(0)], &view(1)).unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        let bytes = u64::from(tokens) * 2048 * 2;
        assert_eq!((cost.read_bytes, cost.write_bytes), (bytes, bytes));
        assert_eq!(cost.metadata_read_bytes, 0);
    }
}
