#[forja_sdk::kernel(map)]
fn closure(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    (|value| value)(x)
}

fn main() {}
