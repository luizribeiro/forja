use std::mem::{MaybeUninit, size_of};

use mach2::{
    kern_return::KERN_SUCCESS,
    task::task_info,
    task_info::{TASK_VM_INFO, task_info_t, task_vm_info},
    traps::mach_task_self,
    vm_types::natural_t,
};

pub(super) fn physical_footprint() -> Option<(u64, Option<u64>)> {
    let mut info = MaybeUninit::<task_vm_info>::zeroed();
    let mut count = u32::try_from(size_of::<task_vm_info>() / size_of::<natural_t>()).ok()?;
    // SAFETY: `info` is writable for `count` natural words, and both pointers remain live for the
    // synchronous Mach call. A successful call initializes the requested task information.
    let result = unsafe {
        task_info(
            mach_task_self(),
            TASK_VM_INFO,
            info.as_mut_ptr().cast::<natural_t>() as task_info_t,
            &raw mut count,
        )
    };
    if result != KERN_SUCCESS {
        return None;
    }
    // SAFETY: `task_info` returned success after writing the requested `task_vm_info` record.
    let info = unsafe { info.assume_init() };
    Some(footprint_readings(
        info.phys_footprint,
        info.ledger_phys_footprint_peak,
    ))
}

fn footprint_readings(current: u64, peak: i64) -> (u64, Option<u64>) {
    (current, u64::try_from(peak).ok())
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_the_process_physical_footprint() {
        let (current, peak) = super::physical_footprint().unwrap();
        assert!(current > 0);
        assert!(peak.is_none_or(|peak| peak >= current));
    }

    #[test]
    fn invalid_peak_preserves_the_current_footprint() {
        assert_eq!(super::footprint_readings(17, -1), (17, None));
    }
}
