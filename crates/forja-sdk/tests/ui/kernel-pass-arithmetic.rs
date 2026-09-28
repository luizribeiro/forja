use forja_sdk::Tensor;

const OFFSET: f32 = 0.25;

#[forja_sdk::kernel(map)]
fn arithmetic(
    x: forja_sdk::kernel::Elem,
    y: forja_sdk::kernel::Elem,
    scale: f32,
    variant: u32,
) -> forja_sdk::kernel::Elem {
    let value: f32 = (x + y) * scale;
    let value = {
        let shifted = value - (OFFSET + 0.5);
        -shifted / 2.0
    };
    value + x
}

fn check_calls() -> forja_sdk::Result<()> {
    let x = Tensor::from_slice(&[1.0_f32], &[1])?;
    let y = Tensor::from_slice(&[2.0_f32], &[1])?;
    let out = Tensor::<f32>::zeros(&[1])?;
    let _ = arithmetic(&x, &y, 0.5, 1)?;
    arithmetic_into(&x, &y, 0.5, 1, &out)
}

fn main() {}
