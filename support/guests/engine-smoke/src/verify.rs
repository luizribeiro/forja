use super::{BatchIn, BatchOut, BatchPhase, Error, Tensor, f32_tensor, u32_tensor};
use crate::exports::l9o::gpu::engine::SamplingParams;
use crate::l9o::gpu::{
    compute::{Affine, CommandList, Dtype, Op, SampleCfg, submit},
    paged_storage::{
        BatchSlot, CopyDirection, PagePool, PagedSpan, StateInputLayout, StateOperation,
        StatePlaneDescriptor, record_feedback, record_paged_copy, record_state,
    },
};

pub(super) fn open(pool: &PagePool) -> Result<BatchSlot, Error> {
    pool.open_state_slot(
        1,
        &[StatePlaneDescriptor {
            dtype: Dtype::F32,
            shape: vec![3, 1],
        }],
        StateInputLayout {
            vocab: 4,
            prompt_tokens: 16,
            verify_rows: 3,
        },
    )
}

pub(super) async fn run(slot: &BatchSlot, input: BatchIn) -> Result<BatchOut, Error> {
    let verify = input.shape.phase == BatchPhase::Verify;
    let commands = CommandList::new();
    let mut outputs = Vec::new();
    for lane in input.lanes {
        let Some(lane) = lane else {
            outputs.push(None);
            continue;
        };
        let tokens = if let Some(tokens) = lane.tokens {
            tokens
        } else {
            let bytes = slot.token(lane.slot)?.read().await?;
            vec![u32::from_le_bytes(bytes.try_into().map_err(|_| {
                Error::Layout("feedback is not scalar".to_owned())
            })?)]
        };
        let count = u32::try_from(tokens.len())
            .map_err(|_| Error::Layout("token extent exceeds u32".to_owned()))?;
        append(&commands, slot, lane.slot, lane.start_pos, count)?;
        let rows: &[u32] = if verify {
            &tokens
        } else {
            &tokens[tokens.len() - 1..]
        };
        let start = if verify {
            lane.start_pos
        } else {
            lane.start_pos
                .checked_add(count - 1)
                .ok_or_else(|| Error::Layout("sample position overflow".to_owned()))?
        };
        for (row, &token) in rows.iter().enumerate() {
            let position = start
                .checked_add(
                    u32::try_from(row).map_err(|_| Error::Layout("row exceeds u32".to_owned()))?,
                )
                .ok_or_else(|| Error::Layout("row position overflow".to_owned()))?;
            if token == 3 && position == 3 {
                return Err(Error::Layout("injected batch failure".to_owned()));
            }
            let width = if verify { 1 } else { count };
            let scalar = f32::from(
                u16::try_from(token).map_err(|_| Error::Layout("token exceeds u16".to_owned()))?,
            );
            let count = usize::try_from(width)
                .map_err(|_| Error::Layout("state width exceeds usize".to_owned()))?;
            let value = f32_tensor(&[width, 1], &vec![scalar; count])?;
            let weights = f32_tensor(&[4, 1], &[1.0; 4])?;
            let carry = Tensor::alloc(Dtype::F32, &[width, 1])?;
            record_state(
                &commands,
                slot,
                lane.slot,
                0,
                StateOperation::CausalConvolution,
                &[&value, &weights],
                &carry,
                "",
            )?;
            let (logits, selected) = sample(&commands, token, position, &lane.sampling)?;
            if !verify {
                record_feedback(&commands, slot, lane.slot, &selected)?;
            }
            outputs.push(Some((logits, selected)));
        }
    }
    submit(commands).await?;
    let mut logits = Vec::new();
    let mut tokens = Vec::new();
    for output in outputs {
        if let Some((row, token)) = output {
            logits.extend(row.read().await?);
            tokens.extend(token.read().await?);
        } else {
            logits.extend([0; 16]);
            tokens.extend([0; 4]);
        }
    }
    let rows = if verify {
        input.shape.tokens
    } else {
        input.shape.bucket
    };
    let logit_output = Tensor::alloc(Dtype::F32, &[rows, 4])?;
    logit_output.write(&logits)?;
    let token_output = Tensor::alloc(Dtype::U32, &[rows])?;
    token_output.write(&tokens)?;
    Ok(BatchOut {
        logits: logit_output,
        tokens: token_output,
    })
}

fn append(
    commands: &CommandList,
    slot: &BatchSlot,
    lane: u32,
    start: u32,
    count: u32,
) -> Result<(), Error> {
    let kv = Tensor::alloc(Dtype::F32, &[count, 2])?;
    record_paged_copy(
        commands,
        slot,
        lane,
        PagedSpan {
            plane: 0,
            start: Affine {
                param: None,
                scale: 0,
                offset: start,
            },
            len: u64::from(count),
        },
        &kv,
        CopyDirection::IntoPages,
    )?;
    Ok(())
}

fn sample(
    commands: &CommandList,
    token: u32,
    position: u32,
    sampling: &SamplingParams,
) -> Result<(Tensor, Tensor), Error> {
    let mut values = [1.0; 4];
    values[usize::try_from((token + 1) % 4)
        .map_err(|_| Error::Layout("token exceeds usize".to_owned()))?] = 4.0;
    let logits = f32_tensor(&[1, 4], &values)?;
    let seed = sampling.seed.to_le_bytes();
    let parameters = u32_tensor(
        &[5],
        &[
            sampling.temperature.to_bits(),
            sampling.top_k,
            sampling.top_p.to_bits(),
            u32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]),
            u32::from_le_bytes([seed[4], seed[5], seed[6], seed[7]]),
        ],
    )?;
    let selected = Tensor::alloc(Dtype::U32, &[1])?;
    commands.dispatch(
        Op::Sample(SampleCfg {
            position: Affine {
                param: None,
                scale: 0,
                offset: position,
            },
            chunk_width: 2048,
            fast_top_k: 20,
            fast_max_chunks: 102,
            rejection_rounds: 2,
        }),
        &[&logits, &parameters],
        &selected,
    )?;
    Ok((logits, selected))
}
