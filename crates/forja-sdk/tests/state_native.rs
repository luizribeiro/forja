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
