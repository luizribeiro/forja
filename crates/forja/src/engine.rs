use std::{
    error::Error,
    num::TryFromIntError,
    path::{Path, PathBuf},
    time::Duration,
};

use forja_config::{GraphReplay, Limits as ConfigLimits, Unbounded};

struct HostLimits(forja_host::Limits);

impl TryFrom<&ConfigLimits> for HostLimits {
    type Error = TryFromIntError;

    fn try_from(limits: &ConfigLimits) -> Result<Self, Self::Error> {
        let dispatches_per_list = match limits.dispatches_per_list {
            Unbounded::Unlimited => usize::MAX,
            Unbounded::Limited(value) => usize::try_from(value)?,
        };
        let work_per_dispatch = match limits.work_per_dispatch {
            Unbounded::Unlimited => u64::MAX,
            Unbounded::Limited(value) => value,
        };
        let gpu_time_budget = match limits.gpu_time_budget {
            Unbounded::Unlimited => Duration::MAX,
            Unbounded::Limited(value) => value.get(),
        };
        Ok(Self(
            forja_host::Limits::new(
                limits.live_bytes.get(),
                usize::try_from(limits.tensor_rank)?,
                limits.tensor_elements,
                usize::try_from(limits.live_tensor_handles)?,
                limits.read_bytes.get(),
            )
            .with_kernel_limit(usize::try_from(limits.live_kernels)?)
            .with_graph_limit(usize::try_from(limits.live_graphs)?)
            .with_store_limits(
                usize::try_from(limits.guest_memory_bytes.get())?,
                usize::try_from(limits.table_elements)?,
                usize::try_from(limits.instances)?,
            )
            .with_command_limits(dispatches_per_list, work_per_dispatch)
            .with_guest_call_timeout(limits.guest_call_timeout.get())
            .with_gpu_limits(limits.submission_timeout.get(), gpu_time_budget),
        ))
    }
}

/// Selects the last greatest value under the IEEE total order.
///
/// Equal values select the later index, and positive NaNs sort above finite values rather than
/// being ignored.
pub(crate) fn argmax(values: &[f32]) -> Result<u32, Box<dyn Error>> {
    let index = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or("cannot take argmax of empty logits")?;
    Ok(u32::try_from(index)?)
}

pub(crate) fn read_token(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "selected token must contain exactly four bytes")?;
    Ok(u32::from_le_bytes(bytes))
}

pub(crate) fn limits(config: &ConfigLimits) -> Result<forja_host::Limits, TryFromIntError> {
    HostLimits::try_from(config).map(|limits| limits.0)
}

pub(crate) fn weights_path(model_dir: &Path) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(model_dir).map_err(|error| {
        format!(
            "cannot resolve model directory {}: {error}",
            model_dir.display()
        )
    })?;
    for name in ["model.safetensors", "model.safetensors.index.json"] {
        let candidate = model_dir.join(name);
        if !candidate.try_exists().map_err(|error| {
            format!(
                "cannot inspect model weights {}: {error}",
                candidate.display()
            )
        })? {
            continue;
        }
        let resolved = std::fs::canonicalize(&candidate).map_err(|error| {
            format!(
                "cannot resolve model weights {}: {error}",
                candidate.display()
            )
        })?;
        if !resolved.starts_with(&root) {
            return Err(format!(
                "model weights resolve outside the model directory: {}",
                candidate.display()
            ));
        }
        if !resolved.is_file() {
            return Err(format!(
                "model weights are not a file: {}",
                candidate.display()
            ));
        }
        return Ok(resolved);
    }
    Err(format!(
        "model directory has no model.safetensors or model.safetensors.index.json: {}",
        model_dir.display()
    ))
}

#[cfg(target_os = "macos")]
pub(crate) const fn metal_graph_replay(value: GraphReplay) -> forja_metal::MetalGraphReplay {
    match value {
        GraphReplay::Tier1 => forja_metal::MetalGraphReplay::Tier1,
        GraphReplay::Tier2 => forja_metal::MetalGraphReplay::Tier2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, time::SystemTime, time::UNIX_EPOCH};

    #[test]
    fn argmax_rejects_empty_values() {
        assert_eq!(
            argmax(&[]).unwrap_err().to_string(),
            "cannot take argmax of empty logits"
        );
    }

    #[test]
    fn builds_the_existing_host_limits_from_config() {
        let expected = forja_host::Limits::new(
            48 * 1024 * 1024 * 1024,
            4,
            1_000_000_000,
            20_000,
            1024 * 1024 * 1024,
        )
        .with_command_limits(4_096, u64::MAX)
        .with_guest_call_timeout(Duration::from_secs(300))
        .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600));
        assert_eq!(limits(&ConfigLimits::default()).unwrap(), expected);
    }

    #[test]
    fn argmax_selects_the_last_tied_value() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0])?, 2);
        Ok(())
    }

    #[test]
    fn argmax_uses_total_order_for_nan() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0])?, 1);
        Ok(())
    }

    #[test]
    fn reads_one_selected_token() -> Result<(), Box<dyn Error>> {
        assert_eq!(read_token(&7_u32.to_le_bytes())?, 7);
        assert!(read_token(&[0; 8]).is_err());
        Ok(())
    }

    #[test]
    fn selects_single_or_indexed_model_weights() -> Result<(), Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-weights-path-{nonce}"));
        fs::create_dir(&root)?;
        let index = root.join("model.safetensors.index.json");
        fs::write(&index, b"index")?;
        assert_eq!(weights_path(&root)?, fs::canonicalize(index)?);
        let single = root.join("model.safetensors");
        fs::write(&single, b"single")?;
        assert_eq!(weights_path(&root)?, fs::canonicalize(single)?);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[cfg(unix)]
    fn assert_outside_weight_symlink_is_rejected(name: &str) -> Result<(), Box<dyn Error>> {
        use std::os::unix::fs::symlink;

        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!("forja-weights-root-{name}-{nonce}"));
        let outside = std::env::temp_dir().join(format!("forja-weights-outside-{name}-{nonce}"));
        fs::create_dir(&root)?;
        fs::create_dir(&outside)?;
        let target = outside.join("weights");
        fs::write(&target, b"weights")?;
        symlink(target, root.join(name))?;
        assert!(weights_path(&root).unwrap_err().contains("outside"));
        fs::remove_dir_all(root)?;
        fs::remove_dir_all(outside)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_single_file_weights_symlinked_outside() -> Result<(), Box<dyn Error>> {
        assert_outside_weight_symlink_is_rejected("model.safetensors")
    }

    #[cfg(unix)]
    #[test]
    fn rejects_weight_indexes_symlinked_outside() -> Result<(), Box<dyn Error>> {
        assert_outside_weight_symlink_is_rejected("model.safetensors.index.json")
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn maps_resolved_graph_replay_to_metal() {
        assert_eq!(
            metal_graph_replay(GraphReplay::Tier1),
            forja_metal::MetalGraphReplay::Tier1
        );
        assert_eq!(
            metal_graph_replay(GraphReplay::Tier2),
            forja_metal::MetalGraphReplay::Tier2
        );
    }
}
