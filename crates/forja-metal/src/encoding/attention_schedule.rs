use super::{
    Dispatch, MetalVariant, SdpaKernel, paged_gqa_blocks, paged_gqa_first_pass, select_sdpa,
    selected_variant, steel_attention_tile, vector_two_pass_blocks,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttentionSchedule {
    Decomposed,
    Tiled {
        query: u32,
        key: u32,
    },
    Vector {
        blocks: u32,
        heads_per_group: u32,
        reuse: u32,
    },
}
impl AttentionSchedule {
    pub(crate) fn selected(d: &Dispatch) -> Option<Self> {
        if matches!(d.op(), forja_core::Op::Sdpa { .. }) {
            return Self::contiguous(d, select_sdpa(d).ok()?);
        }
        let [query, key, ..] = d.paged_attention_inputs()?;
        let heads = *query.layout().shape().first()?;
        let kv_heads = *key.layout().shape().get(2)?;
        let width = *query.layout().shape().last()?;
        let tiled = match selected_variant(d)? {
            MetalVariant::PagedSdpaDecomposed => return Some(Self::Decomposed),
            MetalVariant::PagedSdpaSteel | MetalVariant::PagedSdpaSteel256 => {
                Some(SdpaKernel::Steel)
            }
            MetalVariant::PagedSdpaSteelHalf | MetalVariant::PagedSdpaSteelHalf256 => {
                Some(SdpaKernel::SteelHalf)
            }
            MetalVariant::PagedSdpaSteelHalf32x16 => {
                return Some(Self::Tiled { query: 32, key: 16 });
            }
            MetalVariant::PagedSdpaSteelHalf64x16 => {
                return Some(Self::Tiled { query: 64, key: 16 });
            }
            _ => None,
        };
        if let Some(kernel) = tiled {
            let (query, key) = steel_attention_tile(kernel, width).ok()?;
            return Some(Self::Tiled { query, key });
        }
        let blocks = paged_gqa_blocks(selected_variant(d)).unwrap_or(1);
        let heads_per_group = if blocks == 1 || width == 256 {
            1
        } else {
            let (_, _, pack) = paged_gqa_first_pass(heads, kv_heads).ok()?;
            heads.checked_div(kv_heads)?.checked_mul(pack)?
        };
        Some(Self::Vector {
            blocks,
            heads_per_group,
            reuse: 1,
        })
    }
    pub(super) fn contiguous(d: &Dispatch, kernel: SdpaKernel) -> Option<Self> {
        let input = d.primitive_inputs();
        let [heads, queries, width]: [u32; 3] = input.first()?.layout().shape().try_into().ok()?;
        Some(match kernel {
            SdpaKernel::Decomposed => Self::Decomposed,
            SdpaKernel::Steel | SdpaKernel::SteelHalf => {
                let (query, key) = steel_attention_tile(kernel, width).ok()?;
                Self::Tiled { query, key }
            }
            SdpaKernel::Vector | SdpaKernel::VectorGqa => {
                let blocks = vector_two_pass_blocks(d).unwrap_or(1);
                let ratio = heads.checked_div(*input.get(1)?.layout().shape().first()?)?;
                Self::Vector {
                    blocks,
                    heads_per_group: if blocks == 1 || width == 256 {
                        1
                    } else {
                        ratio.checked_mul(queries)?
                    },
                    reuse: if kernel == SdpaKernel::VectorGqa && ratio == 8 && blocks > 1 {
                        2
                    } else {
                        1
                    },
                }
            }
        })
    }
}
