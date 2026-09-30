//! Differential coverage for row-wise top-k selection.

use forja_core::{Backend, CommandList, DType, Op, Submission};
use forja_cpu::CpuBackend;
use forja_metal::MetalBackend;

#[test]
fn top_k_matches_cpu_for_router_and_edge_widths() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    for width in [1_u32, 7, 33, 64, 128, 2050, 151_936] {
        for k in [1_u32, 8, 64].into_iter().filter(|&k| k <= width) {
            let values = (0..width)
                .map(|index| {
                    f32::from(u16::try_from(index.wrapping_mul(17) % 251).unwrap()) - 125.0
                })
                .collect::<Vec<_>>();
            assert_case(&cpu, &metal, DType::F32, &values, width, k);
        }
    }
}

#[test]
fn top_k_orders_ties_and_nan_payloads_deterministically() {
    let cpu = CpuBackend::new();
    let metal = MetalBackend::new().unwrap();
    let mut values = (0_u16..128)
        .map(|index| f32::from(index % 4))
        .collect::<Vec<_>>();
    values[3] = f32::from_bits(0x7fc0_0001);
    values[9] = f32::from_bits(0x7fc0_0002);
    values[17] = f32::from_bits(0xffc0_0001);
    assert_case(&cpu, &metal, DType::F32, &values, 128, 64);
}

fn assert_case(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    dtype: DType,
    values: &[f32],
    width: u32,
    k: u32,
) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let expected = run(cpu, dtype, &bytes, width, k);
    let actual = run(metal, dtype, &bytes, width, k);
    assert_eq!(actual, expected);
}

fn run<B: Backend>(
    backend: &B,
    dtype: DType,
    bytes: &[u8],
    width: u32,
    k: u32,
) -> (Vec<u8>, Vec<u8>) {
    let input = backend.alloc(dtype, &[1, width]).unwrap();
    backend.write(&input, bytes).unwrap();
    let values = backend.alloc(dtype, &[1, k]).unwrap();
    let indices = backend.alloc(DType::U32, &[1, k]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch_many(Op::TopK { k }, &[&input], &[&values, &indices])
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    (
        backend.read(&values).unwrap(),
        backend.read(&indices).unwrap(),
    )
}
