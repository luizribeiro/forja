#[forja_sdk::kernel(row)]
fn row_expressions(x: forja_sdk::kernel::Row) -> forja_sdk::kernel::Row {
    let lane = forja_sdk::kernel::index(-1);
    let extent = forja_sdk::kernel::extent(-1).min(x.len());
    let active = if lane < extent { x } else { 0.0 };
    active.row_sum() + active.row_max() + active.row_min() + active.row_mean()
}

fn main() {}
