//! Minimal hand-written engine component used to test the host contract.

#![allow(clippy::same_length_and_capacity, clippy::unused_async_trait_impl)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "engine-component",
});

use exports::l9o::gpu::engine::{
    BatchIn, BatchOut, BatchPhase, BatchSelection, BatchShape, CachePlane, DecodeIn, DecodeOut,
    EngineInfo, Guest, LoadConfig, ServingInfo, StepIn, StepOut,
};
use l9o::gpu::compute::{Affine, CommandList, Dtype, Error, Tensor, Weights, submit};
use l9o::gpu::paged_storage::{CopyDirection, PagedSpan, record_paged_copy};

thread_local! {
    static PAGED_SLOT: std::cell::RefCell<Option<l9o::gpu::paged_storage::BatchSlot>> =
        const { std::cell::RefCell::new(None) };
    static FEEDBACK: std::cell::RefCell<[u32; 2]> = const { std::cell::RefCell::new([0; 2]) };
}

struct Component;

impl Guest for Component {
    fn describe() -> EngineInfo {
        EngineInfo {
            prompt_block_tokens: None,
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
                prefill_chunk_multiple: Some(2),
                batch_shapes: [1, 2]
                    .into_iter()
                    .flat_map(|bucket| {
                        [
                            BatchSelection::Combined,
                            BatchSelection::Forward,
                            BatchSelection::Masked,
                        ]
                        .map(move |selection| BatchShape {
                            phase: BatchPhase::Decode,
                            bucket,
                            tokens: 1,
                            selection,
                        })
                    })
                    .chain([BatchShape {
                        phase: BatchPhase::Prompt,
                        bucket: 1,
                        tokens: 16,
                        selection: BatchSelection::Combined,
                    }])
                    .collect(),
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
        FEEDBACK.with(|feedback| *feedback.borrow_mut() = [0; 2]);
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

    async fn begin_decode(
        _tokens: Option<Vec<u32>>,
        _start_pos: u32,
        phase: BatchPhase,
    ) -> Result<Tensor, Error> {
        Err(Error::OpSignature(
            match phase {
                BatchPhase::Prompt => "prompt forward",
                BatchPhase::Decode => "decode forward",
            }
            .to_owned(),
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

    async fn batch_step(
        slot: &l9o::gpu::paged_storage::BatchSlot,
        input: BatchIn,
    ) -> Result<BatchOut, Error> {
        if input.lanes.iter().flatten().any(|lane| {
            lane.tokens
                .as_ref()
                .and_then(|tokens| tokens.first())
                .copied()
                == Some(u32::MAX)
        }) {
            loop {
                std::hint::spin_loop();
            }
        }
        touch_paged_slot(slot, &input).await?;
        batch_output(input)
    }

    async fn batch_forward(
        slot: &l9o::gpu::paged_storage::BatchSlot,
        input: exports::l9o::gpu::engine::BatchIn,
    ) -> Result<Tensor, Error> {
        touch_paged_slot(slot, &input).await?;
        Ok(batch_output(input)?.logits)
    }

    async fn batch_select(
        _slot: &l9o::gpu::paged_storage::BatchSlot,
        _logits: &Tensor,
        input: exports::l9o::gpu::engine::BatchSelectIn,
    ) -> Result<exports::l9o::gpu::engine::BatchOut, Error> {
        let bucket = input.shape.bucket;
        let tokens = input
            .lanes
            .into_iter()
            .map(|lane| {
                lane.and_then(|lane| {
                    lane.allow.iter().enumerate().find_map(|(word, &bits)| {
                        (bits != 0).then(|| {
                            u32::try_from(word)
                                .unwrap_or(u32::MAX)
                                .saturating_mul(32)
                                .saturating_add(bits.trailing_zeros())
                        })
                    })
                })
                .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        let logits = vec![0.0; usize::try_from(bucket).unwrap_or(0) * 4];
        Ok(BatchOut {
            logits: f32_tensor(&[bucket, 4], &logits)?,
            tokens: u32_tensor(&[bucket], &tokens)?,
        })
    }
}

async fn touch_paged_slot(
    slot: &l9o::gpu::paged_storage::BatchSlot,
    input: &BatchIn,
) -> Result<(), Error> {
    let commands = CommandList::new();
    let mut values = Vec::new();
    for lane in input.lanes.iter().flatten() {
        let tokens = lane
            .tokens
            .as_ref()
            .map_or(1, |tokens| u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        let value = Tensor::alloc(Dtype::F32, &[tokens, 2])?;
        value.write(&vec![0; usize::try_from(tokens).unwrap_or(0) * 8])?;
        record_paged_copy(
            &commands,
            slot,
            lane.slot,
            PagedSpan {
                plane: 0,
                start: Affine {
                    param: None,
                    scale: 0,
                    offset: lane.start_pos,
                },
                len: u64::from(tokens),
            },
            &value,
            CopyDirection::IntoPages,
        )?;
        values.push(value);
    }
    submit(commands).await?;
    Ok(())
}

fn batch_output(input: BatchIn) -> Result<BatchOut, Error> {
    let corrupt = input.lanes.iter().flatten().any(|lane| {
        lane.tokens
            .as_ref()
            .and_then(|tokens| tokens.first())
            .copied()
            == Some(100)
    });
    let rows = usize::try_from(input.shape.bucket)
        .map_err(|_| Error::Layout("batch bucket exceeds usize".to_owned()))?;
    let mut logits = Vec::with_capacity(rows * 4);
    let mut tokens = Vec::with_capacity(rows);
    for lane in input.lanes {
        let Some(lane) = lane else {
            logits.extend([0.0; 4]);
            tokens.push(0);
            continue;
        };
        let slot = usize::try_from(lane.slot)
            .map_err(|_| Error::Layout("batch slot exceeds usize".to_owned()))?;
        let token = lane
            .tokens
            .as_ref()
            .and_then(|tokens| tokens.last())
            .copied()
            .unwrap_or_else(|| FEEDBACK.with(|feedback| feedback.borrow()[slot]));
        if input.shape.selection == BatchSelection::Combined && token == 3 && lane.start_pos == 3 {
            return Err(Error::Layout("injected batch failure".to_owned()));
        }
        let selected = (token + 1) % 4;
        FEEDBACK.with(|feedback| feedback.borrow_mut()[slot] = selected);
        let mut row = [1.0; 4];
        row[usize::try_from(selected)
            .map_err(|_| Error::Layout("token does not fit usize".to_owned()))?] = 4.0;
        logits.extend(row);
        tokens.push(selected);
    }
    let output_rows = if corrupt { 1_usize } else { rows };
    let output_shape = [
        u32::try_from(output_rows)
            .map_err(|_| Error::Layout("batch output rows exceed u32".to_owned()))?,
        4,
    ];
    logits.truncate(output_rows * 4);
    Ok(BatchOut {
        logits: f32_tensor(&output_shape, &logits)?,
        tokens: u32_tensor(&[input.shape.bucket], &tokens)?,
    })
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

fn u32_tensor(shape: &[u32], values: &[u32]) -> Result<Tensor, Error> {
    let tensor = Tensor::alloc(Dtype::U32, shape)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    tensor.write(&bytes)?;
    Ok(tensor)
}

export!(Component);
