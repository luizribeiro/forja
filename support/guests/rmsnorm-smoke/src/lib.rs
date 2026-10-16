//! Component exercising command recording, submission, and tensor retention.

#![allow(
    clippy::needless_pass_by_value,
    clippy::same_length_and_capacity,
    clippy::unused_async_trait_impl
)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "rmsnorm-smoke",
});

use l9o::gpu::compute::{
    Binop, CommandList, Dtype, Inst, Kernel, KernelSignature, Op, ProgramKind, ProgramSource,
    Tensor, Unop, submit,
};

struct Component;

impl Guest for Component {
    async fn run() -> Result<(u64, u64), String> {
        let input = Tensor::alloc(Dtype::F32, &[7, 1024]).map_err(error)?;
        let weight = Tensor::alloc(Dtype::F32, &[1024]).map_err(error)?;
        let normalized = Tensor::alloc(Dtype::F32, &[7, 1024]).map_err(error)?;
        let activated = Tensor::alloc(Dtype::F32, &[7, 1024]).map_err(error)?;
        input.write(&input_bytes()).map_err(error)?;
        weight.write(&weight_bytes()).map_err(error)?;

        let commands = CommandList::new();
        commands
            .dispatch(Op::RmsNorm(0.00001), &[&input, &weight], &normalized)
            .map_err(error)?;
        let silu_mul = silu_mul_kernel().map_err(error)?;
        commands
            .dispatch_kernel(&silu_mul, &[&normalized, &input], &[&activated])
            .map_err(error)?;
        drop(input);
        drop(weight);
        submit(commands).await.map_err(error)?;

        let normalized = normalized.read().await.map_err(error)?;
        let activated = activated.read().await.map_err(error)?;
        Ok((checksum(&normalized), checksum(&activated)))
    }
}

fn silu_mul_kernel() -> Result<Kernel, l9o::gpu::compute::Error> {
    Kernel::create(
        &ProgramSource {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Unary((Unop::Sigmoid, 0)),
                Inst::Binary((Binop::Mul, 0, 1)),
                Inst::Input(1),
                Inst::Binary((Binop::Mul, 2, 3)),
            ],
            outputs: vec![(0, 4)],
        },
        &KernelSignature {
            rank: 2,
            inputs: vec![Dtype::F32, Dtype::F32],
            outputs: vec![Dtype::F32],
            scalars: 0,
        },
    )
}

fn input_bytes() -> Vec<u8> {
    (0_u16..7 * 1024)
        .flat_map(|index| ((f32::from(index % 257) - 128.0) / 37.0).to_le_bytes())
        .collect()
}

fn weight_bytes() -> Vec<u8> {
    (0_u16..1024)
        .flat_map(|index| (0.5 + f32::from(index) / 2048.0).to_le_bytes())
        .collect()
}

fn error(error: l9o::gpu::compute::Error) -> String {
    format!("{error:?}")
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

export!(Component);
