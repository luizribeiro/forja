use forja_sdk::{DType, Tensor, bf16, f16};

#[forja_sdk::kernel(row)]
fn row_map(x: forja_sdk::kernel::Row, bias: f32) -> forja_sdk::kernel::Row {
    x
}

#[forja_sdk::kernel(map)]
fn pair(
    x: forja_sdk::kernel::Elem,
    y: forja_sdk::kernel::Elem,
) -> (forja_sdk::kernel::Elem, forja_sdk::kernel::Elem) {
    (x, y)
}

#[forja_sdk::kernel(map)]
fn hygienic(
    __forja_context: forja_sdk::kernel::Elem,
    rank: f32,
    input_dtypes: u32,
    out: f32,
) -> forja_sdk::kernel::Elem {
    __forja_context
}

fn check_calls() -> forja_sdk::Result<()> {
    let x = Tensor::<f16>::zeros(&[1, 7])?;
    let y = Tensor::<bf16>::zeros(&[7])?;
    let first = Tensor::<f32>::zeros(&[1, 7])?;
    let second = Tensor::<f16>::zeros(&[1, 7])?;
    let _ = row_map(&x, 0.125)?;
    let _ = pair(&x, &y)?;
    pair_into(&x, &y.broadcast_as(&[1, 7])?, (&first, &second))?;
    let _ = pair_program(
        2,
        &[DType::F16, DType::BF16],
        &[DType::F32, DType::F16],
    )?;
    Ok(())
}

fn main() {}
