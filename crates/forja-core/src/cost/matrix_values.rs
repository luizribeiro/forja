use super::{CostModel, KernelCost, tests::tensor};
use crate::{CommandList, DType, Op};

#[test]
fn gathered_dense_products_count_selected_matrices() {
    let input = tensor(1, DType::F32, &[7, 1024]);
    let weights = tensor(2, DType::F32, &[16, 1024, 33]);
    let indices = tensor(3, DType::U32, &[7, 8]);
    let output = tensor(4, DType::F32, &[7, 8, 33]);
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::GatherMatmul, &[&input, &weights, &indices], &output)
        .unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(cost.flops, 3_784_704);
    assert_eq!(cost.weight_bytes, 7_569_408);
    assert_eq!((cost.read_bytes, cost.write_bytes), (7_598_304, 7392));
    assert_eq!(
        (cost.minimum_weight_bytes, cost.maximum_weight_bytes),
        (1_081_344, 2_162_688)
    );
}

#[test]
fn quantized_embedding_rows_include_affine_metadata() {
    let packed = tensor(1, DType::U32, &[4097, 128]);
    let scales = tensor(2, DType::BF16, &[4097, 16]);
    let biases = tensor(3, DType::BF16, &[4097, 16]);
    let ids = tensor(4, DType::U32, &[7]);
    let output = tensor(5, DType::F32, &[7, 1024]);
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::QuantEmbed {
                bits: 4,
                group_size: 64,
            },
            &[&packed, &scales, &biases, &ids],
            &output,
        )
        .unwrap();
    let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
    assert_eq!(cost.weight_bytes, 4032);
    assert_eq!((cost.read_bytes, cost.write_bytes), (4060, 28_672));
    assert_eq!(cost.flops, 0);
}

#[test]
fn grouped_tiling_bounds_include_both_projection_arms() {
    let input = tensor(1, DType::F32, &[7, 1024]);
    let packed = tensor(2, DType::U32, &[16, 33, 128]);
    let scales = tensor(3, DType::BF16, &[16, 33, 16]);
    let biases = tensor(4, DType::BF16, &[16, 33, 16]);
    let indices = tensor(5, DType::U32, &[7, 8]);
    let output = tensor(6, DType::F32, &[7, 8, 33]);
    let second_packed = tensor(7, DType::U32, &[16, 33, 128]);
    let second_scales = tensor(8, DType::BF16, &[16, 33, 16]);
    let second_biases = tensor(9, DType::BF16, &[16, 33, 16]);
    for (paired, expected) in [
        (false, [8_388_608, 71_303_168]),
        (true, [16_777_216, 142_606_336]),
    ] {
        let mut inputs = vec![&input, &packed, &scales, &biases];
        if paired {
            inputs.extend([&second_packed, &second_scales, &second_biases]);
        }
        inputs.push(&indices);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::GatherQuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &inputs,
                &output,
            )
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        let cost = KernelCost::of(dispatch).unwrap();
        let model = CostModel {
            tile: [32, 32, 32],
            ..CostModel::default()
        };
        assert_eq!(model.padded_flops(dispatch, cost), Some(expected));
        let model = CostModel {
            conditional_tiles: true,
            ..model
        };
        assert_eq!(
            model.padded_flops(dispatch, cost),
            Some([cost.flops, expected[1]])
        );
    }
}
