#[forja_sdk::kernel(map)]
fn assignment(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    let value = x;
    value = x;
    value
}

fn main() {}
