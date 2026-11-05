//! Integration coverage for the Metal backend.

#[path = "../common/mod.rs"]
mod common;
#[path = "../dispatch_safety.rs"]
mod dispatch_safety;
#[path = "../gather_matmul.rs"]
mod gather_matmul;
#[path = "../indirect_commands.rs"]
mod indirect_commands;
#[path = "../map_programs.rs"]
mod map_programs;
#[path = "../nan_comparisons.rs"]
mod nan_comparisons;
#[path = "../quant_embed.rs"]
mod quant_embed;
#[path = "../quant_matmul.rs"]
mod quant_matmul;
#[path = "../queue_ordering.rs"]
mod queue_ordering;
#[path = "../read_only_views.rs"]
mod read_only_views;
#[path = "../rounding.rs"]
mod rounding;
#[path = "../row_programs.rs"]
mod row_programs;
#[path = "../state.rs"]
mod state;
#[path = "../state_block.rs"]
mod state_block;
#[path = "../state_block_grouping.rs"]
mod state_block_grouping;
#[path = "../state_block_reference.rs"]
mod state_block_reference;
#[path = "../state_conv_grouping.rs"]
mod state_conv_grouping;
#[path = "../state_decay.rs"]
mod state_decay;
#[path = "../top_k.rs"]
mod top_k;
#[path = "../variant_dispatch.rs"]
mod variant_dispatch;
