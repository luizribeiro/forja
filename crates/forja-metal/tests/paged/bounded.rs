use super::*;

fn compare(width: u32, length: u32) {
    let expected = reference(width, length);
    let suffix = if width == 256 { "-256" } else { "" };
    let mut variants = vec![
        format!("sdpa.steel-bounded{suffix}"),
        format!("paged-sdpa.steel-bounded{suffix}"),
    ];
    if width == 256 {
        variants.push("paged-sdpa.vector-bounded-256-two-pass".into());
    }
    for variant in variants {
        let float = attend(width, length, 1, DType::F32, &variant);
        let half = attend(width, length, 1, DType::F16, &variant);
        for (dtype, actual) in [(DType::F32, &float), (DType::F16, &half)] {
            let error = forja_testing::normwise_relative_error(&expected, actual);
            assert!(
                error <= forja_testing::dtype_tolerance(dtype).unwrap(),
                "width={width} length={length} variant={variant} dtype={dtype:?} error={error}"
            );
            eprintln!(
                "correctness diagnostic: width={width} length={length} variant={variant} dtype={dtype:?} normwise_error={error}"
            );
        }
        assert!(
            forja_testing::normwise_relative_error(&float, &half)
                <= forja_testing::dtype_tolerance(DType::F16).unwrap()
        );
    }
}

macro_rules! cases {
    ($($name:ident: $width:literal, $length:literal),+ $(,)?) => { $(#[test] fn $name() { compare($width, $length); })+ };
}
cases! {
    metal_bounded_64_position_zero: 64, 1,
    metal_bounded_64_position_63: 64, 64,
    metal_bounded_64_position_64: 64, 65,
    metal_bounded_64_position_32k: 64, 32_769,
    metal_bounded_64_position_128k: 64, 131_073,
    metal_bounded_64_position_256k: 64, 262_144,
    metal_bounded_128_position_zero: 128, 1,
    metal_bounded_128_position_63: 128, 64,
    metal_bounded_128_position_64: 128, 65,
    metal_bounded_128_position_32k: 128, 32_769,
    metal_bounded_128_position_128k: 128, 131_073,
    metal_bounded_128_position_256k: 128, 262_144,
    metal_bounded_256_position_zero: 256, 1,
    metal_bounded_256_position_63: 256, 64,
    metal_bounded_256_position_64: 256, 65,
    metal_bounded_256_position_32k: 256, 32_769,
    metal_bounded_256_position_128k: 256, 131_073,
    metal_bounded_256_position_256k: 256, 262_144,
}

#[test]
fn metal_bounded_prefill_preserves_rows_across_query_batches() {
    for width in [64, 128, 256] {
        let suffix = if width == 256 { "-256" } else { "" };
        let stride = usize::try_from(width).unwrap();
        for prefix in ["sdpa", "paged-sdpa"] {
            let variant = format!("{prefix}.steel-bounded{suffix}");
            let batch = attend(width, 4097, 7, DType::F32, &variant);
            for row in 0..7 {
                let length = 4091 + u32::try_from(row).unwrap();
                let single = attend(width, length, 1, DType::F32, &variant);
                let actual = batch
                    .chunks_exact(7 * stride)
                    .flat_map(|head| &head[row * stride..(row + 1) * stride])
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    single
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{variant} row={row}"
                );
                assert!(
                    forja_testing::normwise_relative_error(&reference(width, length), &actual)
                        <= forja_testing::dtype_tolerance(DType::F32).unwrap()
                );
            }
        }
    }
}
