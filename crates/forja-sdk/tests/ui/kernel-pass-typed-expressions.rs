use forja_sdk::Tensor;

const OFFSET: f32 = 0.25;

#[forja_sdk::kernel(map)]
fn typed_expressions(
    x: forja_sdk::kernel::Elem,
    ids: forja_sdk::kernel::Elem<u32>,
    limit: u32,
) -> forja_sdk::kernel::Elem {
    let integer: u32 = 2;
    let boolean: bool = true;
    let tensor_value: u32 = ids;
    let scalar_value: u32 = limit;
    x + OFFSET
}

fn check_calls() -> forja_sdk::Result<()> {
    let x = Tensor::from_slice(&[1.0_f32], &[1])?;
    let ids = Tensor::from_slice(&[0_u32], &[1])?;
    let _ = typed_expressions(&x, &ids, 7)?;
    Ok(())
}

fn main() {}
