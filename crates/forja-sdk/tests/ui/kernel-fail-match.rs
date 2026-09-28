#[forja_sdk::kernel(map)]
fn matched(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    match x { value => value }
}

fn main() {}
