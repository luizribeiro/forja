#[forja_sdk::kernel(map)]
fn mutable(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    let mut value = x;
    value
}

fn main() {}
