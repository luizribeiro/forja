#[forja_sdk::kernel(map)]
fn bad_clamp(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x.clamp(0.0)
}

fn main() {}
