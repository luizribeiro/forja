use super::{Error, Tensor};
use forja_sdk::{Element, bf16, f16};

const POSITIONS: [u32; 6] = [0, 63, 64, 32_768, 131_072, 262_143];

fn rotate<T: Element>(values: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let input = Tensor::from_slice(values, &[6, 24, 256])?.to_dtype::<T>()?;
    let positions = Tensor::from_slice(&POSITIONS, &[6])?;
    let rotated = input.narrow(2, 0, 64)?.rope(&positions, 10_000_000.0)?;
    let output = input.contiguous()?;
    rotated.copy_into(&mut output.narrow(2, 0, 64)?)?;
    let actual = output.to_dtype::<f32>()?.to_vec()?;
    let original = input.to_dtype::<f32>()?.to_vec()?;
    for (before, after) in original
        .as_chunks::<256>()
        .0
        .iter()
        .zip(actual.as_chunks::<256>().0)
    {
        assert_eq!(&before[64..], &after[64..]);
    }
    Ok(actual)
}

#[test]
fn metal_partial_rope_preserves_integer_long_positions() -> Result<(), Box<dyn Error>> {
    let values = (0_u32..6 * 24 * 256)
        .map(|index| f32::from(u16::try_from((index * 7) % 97).unwrap()) / 64.0 - 0.75)
        .collect::<Vec<_>>();
    let expected = rotate::<f32>(&values)?;
    forja_sdk::set_native_device(forja_sdk::NativeDevice::Metal);
    for (dtype, actual) in [
        (forja_core::DType::F32, rotate::<f32>(&values)?),
        (forja_core::DType::F16, rotate::<f16>(&values)?),
        (forja_core::DType::BF16, rotate::<bf16>(&values)?),
    ] {
        let tolerance = forja_testing::dtype_tolerance(dtype)?;
        for (index, position) in POSITIONS.iter().enumerate() {
            let range = index * 24 * 256..(index + 1) * 24 * 256;
            let error =
                golden_fixtures::normwise_relative_error(&expected[range.clone()], &actual[range])?;
            assert!(
                error <= tolerance,
                "position={position} dtype={dtype:?} error={error}"
            );
            eprintln!(
                "correctness diagnostic: RoPE position={position} dtype={dtype:?} normwise_error={error}"
            );
        }
    }
    Ok(())
}
