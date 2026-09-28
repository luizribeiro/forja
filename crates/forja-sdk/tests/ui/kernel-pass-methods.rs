#[forja_sdk::kernel(map)]
fn methods(
    x: forja_sdk::kernel::Elem,
    y: forja_sdk::kernel::Elem,
) -> (
    forja_sdk::kernel::Elem,
    forja_sdk::kernel::Elem,
    forja_sdk::kernel::Elem,
    forja_sdk::kernel::Elem,
) {
    let direct = x.abs()
        + x.exp()
        + x.ln()
        + x.sqrt()
        + x.rsqrt()
        + x.recip()
        + x.sin()
        + x.cos()
        + x.tanh()
        + x.sigmoid()
        + x.floor();
    let binary = x.maximum(y) + x.minimum(y) + x.powf(y);
    let desugared = x.clamp(-1.0, 1.0) + x.ceil() + x.log2() + x.log10() + x.exp2();
    let powers = x.powi(3) + x.powi(-2);
    (direct, binary, desugared, powers)
}

fn main() {}
