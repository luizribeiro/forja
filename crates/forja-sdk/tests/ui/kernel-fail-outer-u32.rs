const COUNT: u32 = 2;

#[forja_sdk::kernel(map)]
fn outer_u32(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x + COUNT
}

fn main() {}
