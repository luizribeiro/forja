#[forja_sdk::kernel(row, helper)]
fn centered(x: f32) -> f32 {
    x - x.row_mean()
}

#[forja_sdk::kernel(map)]
fn map_calls_row(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    centered(x)
}

fn main() {}
