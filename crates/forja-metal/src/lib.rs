//! Metal 4 acceleration behind a safe Forja API.
//! This is the sole crate permitted to contain the unsafe code required by Objective-C interop.

#[cfg(target_os = "macos")]
mod storage;

#[cfg(target_os = "macos")]
pub use storage::MetalBackend;

#[cfg(target_os = "macos")]
mod platform {
    use std::ptr::NonNull;

    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTL4CommandBuffer, MTL4CommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent,
        MTLGPUFamily, MTLSharedEvent,
    };

    pub(super) fn run() -> Result<(String, bool), String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "Metal has no system default device".to_owned())?;
        let name = device.name().to_string();
        let supports_metal4 = device.supportsFamily(MTLGPUFamily::Metal4);
        let queue = device
            .newMTL4CommandQueue()
            .ok_or_else(|| "Metal could not create an MTL4CommandQueue".to_owned())?;
        let allocator = device
            .newCommandAllocator()
            .ok_or_else(|| "Metal could not create an MTL4CommandAllocator".to_owned())?;
        let command_buffer = device
            .newCommandBuffer()
            .ok_or_else(|| "Metal could not create an MTL4CommandBuffer".to_owned())?;
        let shared_event = device
            .newSharedEvent()
            .ok_or_else(|| "Metal could not create an MTLSharedEvent".to_owned())?;

        command_buffer.beginCommandBufferWithAllocator(&allocator);
        command_buffer.endCommandBuffer();

        let command_buffer: &ProtocolObject<dyn MTL4CommandBuffer> = &command_buffer;
        let mut command_buffers = [NonNull::from(command_buffer)];
        // SAFETY: The pointer refers to one live command buffer and the count exactly matches the array.
        unsafe {
            queue.commit_count(
                NonNull::from(&mut command_buffers[0]),
                command_buffers.len(),
            );
        }
        let shared_event_ref: &ProtocolObject<dyn MTLSharedEvent> = &shared_event;
        let event: &ProtocolObject<dyn MTLEvent> = shared_event_ref.as_ref();
        queue.signalEvent_value(event, 1);

        if !shared_event.waitUntilSignaledValue_timeoutMS(1, 4_500) {
            return Err("Metal 4 command buffer did not complete within 4.5 seconds".to_owned());
        }

        Ok((name, supports_metal4))
    }
}

/// Submits an empty Metal 4 command buffer and returns the device name and Metal 4 support.
///
/// # Errors
///
/// Returns an error if Metal cannot create a required object or the GPU does not complete the
/// submission within 4.5 seconds.
pub fn smoke_test() -> Result<(String, bool), String> {
    #[cfg(target_os = "macos")]
    {
        platform::run()
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Metal is available only on macOS".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    #[test]
    fn submits_empty_metal4_command_buffer() {
        let started = Instant::now();
        let (name, supports_metal4) = super::smoke_test().unwrap();

        assert!(name.contains("Apple"), "unexpected Metal device: {name}");
        assert!(supports_metal4, "{name} does not report Metal 4 support");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "Metal 4 submission exceeded five seconds"
        );
    }
}
