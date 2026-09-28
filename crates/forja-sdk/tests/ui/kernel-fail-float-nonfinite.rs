#[forja_sdk::kernel(map)]
fn float_nonfinite(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x + 1e40
}

fn main() {}
