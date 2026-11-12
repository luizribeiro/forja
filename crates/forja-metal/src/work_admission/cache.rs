use super::{MetalBackend, WorkCalibration};
use objc2_metal::MTLDevice;
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::PathBuf,
    time::{Duration, SystemTime},
};

pub(super) struct Cache {
    path: PathBuf,
    identity: String,
}
impl Cache {
    pub(super) fn new(backend: &MetalBackend) -> Option<Self> {
        let build = std::process::Command::new("/usr/bin/sw_vers")
            .arg("-buildVersion")
            .output()
            .ok()?;
        if !build.status.success() {
            return None;
        }
        let identity = format!(
            "admission-v2:{}:{:?}:{}{}{}{}{}{}",
            backend.device.registryID(),
            build.stdout,
            crate::storage::shader_source(),
            include_str!("../work_calibration.rs"),
            include_str!("../calibration.rs"),
            include_str!("../bandwidth.rs"),
            include_str!("../work_admission.rs"),
            include_str!("../../../forja-core/src/work_rates.rs")
        );
        let mut hash = DefaultHasher::new();
        identity.hash(&mut hash);
        let path = PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/Caches/forja/work-admission")
            .join(format!("{:016x}", hash.finish()));
        Some(Self { path, identity })
    }
    pub(super) fn read(&self) -> Option<WorkCalibration> {
        let file = std::fs::File::open(&self.path).ok()?;
        if file.metadata().ok()?.modified().ok()?.elapsed().ok()? > Duration::from_secs(3600) {
            return None;
        }
        let mut record = String::new();
        file.take(16 * 1024 * 1024)
            .read_to_string(&mut record)
            .ok()?;
        WorkCalibration::from_cache_record(&record, &self.identity)
    }
    pub(super) fn write(&self, rates: &WorkCalibration) {
        let _ = self.try_write(rates);
    }
    fn try_write(&self, rates: &WorkCalibration) -> std::io::Result<()> {
        let Some(parent) = self.path.parent() else {
            return Ok(());
        };
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
        let temporary = self.path.with_extension(format!(
            "{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let result = (|| {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?;
            file.write_all(rates.cache_record(&self.identity).as_bytes())?;
            std::fs::rename(&temporary, &self.path)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temporary);
        }
        result
    }
}
