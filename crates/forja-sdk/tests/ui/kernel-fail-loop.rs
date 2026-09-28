#[forja_sdk::kernel(map)]
fn looped(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    loop { x }
}

fn main() {}
