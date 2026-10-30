#![cfg(feature = "native")]
//! Opaque state updates flush lazy tensors and refuse capture.
use forja_sdk::{DType, Result, Tensor, capture, state::StatePlane};
#[test]
fn eager_convolution_carries_history_and_refuses_capture() -> Result<()> {
    let mut state = StatePlane::new(DType::F32, &[1, 1])?;
    let x = Tensor::from_slice(&[2.0], &[1, 1])?;
    let x = (&x + &x)?;
    let weights = Tensor::from_slice(&[1.0, 1.0], &[2, 1])?;
    assert_eq!(state.causal_convolution(&x, &weights)?.to_vec()?, [4.0]);
    assert_eq!(state.causal_convolution(&x, &weights)?.to_vec()?, [8.0]);
    assert!(capture(&[], || state.causal_convolution(&x, &weights)).is_err());
    assert!(StatePlane::new(DType::U32, &[1]).is_err());
    Ok(())
}

#[test]
fn eager_delta_block_flushes_inputs_and_carries_private_state() -> Result<()> {
    let mut state = StatePlane::new(DType::F32, &[1, 1, 1])?;
    let q = Tensor::from_slice(&[1.0_f32, 2.0], &[2, 1, 1])?;
    let k = Tensor::from_slice(&[0.5_f32, 0.25], &[2, 1, 1])?;
    let v = Tensor::from_slice(&[2.0_f32, 3.0], &[2, 1, 1])?;
    let g = Tensor::from_slice(&[0.5_f32.ln(); 2], &[2, 1])?;
    let beta = Tensor::from_slice(&[0.25_f32, 0.5], &[2, 1])?;
    let output = state.delta_rule_block(&q, &k, &v, &g, &beta, None)?;
    assert_eq!(output.to_vec()?, [0.25, 127.0 / 128.0]);
    assert!(capture(&[], || state.delta_rule_block(&q, &k, &v, &g, &beta, None)).is_err());
    Ok(())
}

#[test]
fn native_decode_buffers_preserve_sampling_and_feedback() -> Result<()> {
    use forja_sdk::{SamplingParams, nn::blocks::DecodeState};
    let mut decode = DecodeState::new(64, 33)?;
    let sampling = SamplingParams {
        seed: 0x0123_4567_89ab_cdef,
        ..SamplingParams::default()
    };
    decode.select(sampling)?;
    assert_eq!(
        decode.sampling()?.to_vec()?[3..],
        [0x89ab_cdef, 0x0123_4567]
    );
    let selected = Tensor::from_slice(&[7], &[1])?;
    assert_eq!(decode.store_selected(&selected, 3)?.to_vec()?, [7]);
    assert_eq!(decode.token()?.to_vec()?, [7]);
    Ok(())
}
