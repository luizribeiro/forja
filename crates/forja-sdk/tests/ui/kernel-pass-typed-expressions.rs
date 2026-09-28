use forja_sdk::Tensor;

const OFFSET: f32 = 0.25;

#[forja_sdk::kernel(map)]
fn typed_expressions(
    x: forja_sdk::kernel::Elem,
    ids: forja_sdk::kernel::Elem<u32>,
    limit: u32,
) -> forja_sdk::kernel::Elem {
    let wrapped: u32 = ids.wrapping_add(2).wrapping_mul(3).wrapping_sub(1);
    let bounded = wrapped.min(limit).max(1);
    let in_range: bool = bounded < limit;
    let equal: bool = ids == 0;
    x + OFFSET
}

fn check_calls() -> forja_sdk::Result<()> {
    let x = Tensor::from_slice(&[1.0_f32], &[1])?;
    let ids = Tensor::from_slice(&[0_u32], &[1])?;
    let _ = typed_expressions(&x, &ids, 7)?;
    Ok(())
}

fn main() {}
