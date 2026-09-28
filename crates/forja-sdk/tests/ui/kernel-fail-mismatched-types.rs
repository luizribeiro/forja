#[forja_sdk::kernel(map)]
fn mismatched(x: forja_sdk::kernel::Elem, count: u32) -> forja_sdk::kernel::Elem {
    x + count
}

fn main() {}
