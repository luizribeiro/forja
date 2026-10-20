//! Minimal hand-written engine component used to test the host contract.

#![allow(clippy::same_length_and_capacity, clippy::unused_async_trait_impl)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "engine-component",
});

use exports::l9o::gpu::engine::{
    CachePlane, DecodeIn, DecodeOut, EngineInfo, Guest, LoadConfig, ServingInfo, StepIn, StepOut,
};
use l9o::gpu::compute::{Dtype, Error, Tensor, Weights};

thread_local! {
    static PAGED_SLOT: std::cell::RefCell<Option<l9o::gpu::paged_storage::BatchSlot>> =
        const { std::cell::RefCell::new(None) };
}

struct Component;

impl Guest for Component {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 4,
            max_context: 33,
            tap_layers: vec![0],
            router_layers: vec![],
            serving: Some(ServingInfo {
                page_tokens: 4,
                planes: vec![CachePlane {
                    elements_per_token: 2,
                    alignment: 4,
                    label: "cache".to_owned(),
                }],
                active_lanes: 2,
                max_context: 32,
            }),
        }
    }

    async fn load(
        _weights: &Weights,
        pool: Option<&l9o::gpu::paged_storage::PagePool>,
        _config: LoadConfig,
    ) -> Result<(), Error> {
        let slot = pool.map(|pool| pool.open_slot(1)).transpose()?;
        PAGED_SLOT.with(|state| *state.borrow_mut() = slot);
        Ok(())
    }

    async fn step(input: StepIn) -> Result<StepOut, Error> {
        let mode = input.tokens.first().copied().unwrap_or_default();
        if mode == u32::MAX {
            loop {
                std::hint::spin_loop();
            }
        }
        let logits = if mode < 4 {
            let mut values = vec![1.0; 4];
            let next = usize::try_from((mode + 1) % 4)
                .map_err(|_| Error::Layout("token does not fit usize".to_owned()))?;
            values[next] = 4.0;
            f32_tensor(&[4], &values)?
        } else if mode == 100 {
            f32_tensor(&[3], &[1.0, 2.0, 3.0])?
        } else if mode == 103 {
            let tensor = Tensor::alloc(Dtype::U32, &[4])?;
            tensor.write(&[0; 16])?;
            tensor
        } else {
            f32_tensor(&[4], &[1.0, 2.0, 3.0, 4.0])?
        };
        let taps = if input.taps && mode != 101 {
            let (shape, values) = if mode == 102 {
                (vec![2, 1], vec![1.0; 2])
            } else {
                let sequence = u32::try_from(input.tokens.len())
                    .map_err(|_| Error::Layout("token count exceeds u32".to_owned()))?;
                (vec![sequence, 1], vec![1.0; input.tokens.len()])
            };
            vec![f32_tensor(&shape, &values)?]
        } else {
            vec![]
        };
        Ok(StepOut {
            logits,
            taps,
            router_logits: vec![],
        })
    }

    async fn decode(_input: DecodeIn) -> Result<DecodeOut, Error> {
        Err(Error::OpSignature(
            "engine does not support retained-token decode".to_owned(),
        ))
    }

    async fn begin_decode(_tokens: Option<Vec<u32>>, _start_pos: u32) -> Result<Tensor, Error> {
        Err(Error::OpSignature(
            "engine does not support constrained decode".to_owned(),
        ))
    }

    async fn finish_decode(
        _logits: &Tensor,
        _start_pos: u32,
        _sampling: exports::l9o::gpu::engine::SamplingParams,
        _allow: Vec<u32>,
    ) -> Result<DecodeOut, Error> {
        Err(Error::OpSignature(
            "engine does not support constrained decode".to_owned(),
        ))
    }

    async fn rewind_decode(_next_pos: u32) -> Result<(), Error> {
        Err(Error::OpSignature(
            "engine does not support constrained decode".to_owned(),
        ))
    }
}

fn f32_tensor(shape: &[u32], values: &[f32]) -> Result<Tensor, Error> {
    let tensor = Tensor::alloc(Dtype::F32, shape)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    tensor.write(&bytes)?;
    Ok(tensor)
}

export!(Component);
