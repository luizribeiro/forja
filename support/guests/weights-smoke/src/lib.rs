//! Component exercising host-granted weight files.

#![allow(
    clippy::needless_pass_by_value,
    clippy::same_length_and_capacity,
    clippy::unused_async_trait_impl
)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "weights-smoke",
});

use l9o::gpu::compute::{CommandList, Dtype, Op, Tensor, open_weights, submit};

struct Component;

impl Guest for Component {
    async fn run() -> Result<Vec<u8>, String> {
        let weights = open_weights("model").map_err(error)?;
        if weights.names() != ["projection"] {
            return Err("unexpected weight names".to_owned());
        }
        if open_weights("missing").is_ok() {
            return Err("ungranted weight key opened".to_owned());
        }
        let weight = weights.tensor("projection").map_err(error)?;
        let input = Tensor::alloc(Dtype::F32, &[1, 3]).map_err(error)?;
        let output = Tensor::alloc(Dtype::F32, &[1, 2]).map_err(error)?;
        if Tensor::alloc(Dtype::F32, &[1]).is_ok() {
            return Err("weight tensor did not consume a handle".to_owned());
        }
        input
            .write(
                &[1.0_f32, 2.0, 3.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .map_err(error)?;
        let commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&input, &weight], &output)
            .map_err(error)?;
        submit(commands).await.map_err(error)?;
        output.read().await.map_err(error)
    }
}

fn error(error: l9o::gpu::compute::Error) -> String {
    format!("{error:?}")
}

export!(Component);
