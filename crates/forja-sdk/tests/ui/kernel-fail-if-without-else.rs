#[forja_sdk::kernel(map)]
fn if_without_else(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    if x > 0.0 { x }
}

fn main() {}
