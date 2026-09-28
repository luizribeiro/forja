#[forja_sdk::kernel(map)]
fn unused_tensor(
    x: forja_sdk::kernel::Elem,
    w: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    x
}

fn main() {}
