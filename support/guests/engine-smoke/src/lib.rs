//! Minimal hand-written engine component used to test the host contract.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "engine-component",
});

use exports::l9o::gpu::engine::{EngineInfo, Guest, StepIn, StepOut};
use l9o::gpu::compute::{Dtype, Error, Tensor, Weights};

struct Component;

impl Guest for Component {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 4,
            max_context: 33,
            tap_layers: vec![],
        }
    }

    fn load(_weights: &Weights) -> Result<(), Error> {
        Ok(())
    }

    fn step(input: StepIn) -> Result<StepOut, Error> {
        if input.tokens.first() == Some(&u32::MAX) {
            loop {
                std::hint::spin_loop();
            }
        }
        let logits = Tensor::alloc(Dtype::F32, &[4])?;
        let values = [1.0_f32, 2.0, 3.0, 4.0];
        let bytes = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        logits.write(&bytes)?;
        Ok(StepOut {
            logits,
            taps: vec![],
        })
    }
}

export!(Component);
