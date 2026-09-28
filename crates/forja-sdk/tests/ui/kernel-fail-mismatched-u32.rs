#[forja_sdk::kernel(map)]
fn mismatched_u32(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    let count: u32 = x;
    x
}

fn main() {}
