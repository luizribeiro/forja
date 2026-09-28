#[forja_sdk::kernel(map)]
fn unknown_method(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x.tan()
}

fn main() {}
