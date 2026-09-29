//! Cross-submission dependency semantics for a Metal 4 command queue.

#![cfg(target_os = "macos")]

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
    MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLAllocation, MTLBuffer, MTLCompileOptions,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLSize,
};

const SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

kernel void advance(device uint *token [[buffer(0)]], device uint *kv [[buffer(1)]]) {
    kv[0] += token[0];
    token[0] += 1;
}
";

#[test]
fn queue_commits_preserve_token_and_cache_dependencies() {
    const STEPS: usize = 128;

    let device = MTLCreateSystemDefaultDevice().unwrap();
    let queue = device.newMTL4CommandQueue().unwrap();
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(SOURCE),
            Some(&MTLCompileOptions::new()),
        )
        .unwrap();
    let function = library
        .newFunctionWithName(&NSString::from_str("advance"))
        .unwrap();
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap();
    let token = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let kv = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    // SAFETY: Each shared allocation contains one u32 and GPU work has not started.
    unsafe {
        token.contents().cast::<u32>().write(1);
        kv.contents().cast::<u32>().write(0);
    }

    let table_descriptor = MTL4ArgumentTableDescriptor::new();
    table_descriptor.setMaxBufferBindCount(2);
    let table = device
        .newArgumentTableWithDescriptor_error(&table_descriptor)
        .unwrap();
    // SAFETY: Both indices are within the table and the buffers remain live through completion.
    unsafe {
        table.setAddress_atIndex(token.gpuAddress(), 0);
        table.setAddress_atIndex(kv.gpuAddress(), 1);
    }
    let residency = device
        .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
        .unwrap();
    for raw in [&token, &kv] {
        let allocation: &ProtocolObject<dyn MTLAllocation> = raw.as_ref();
        residency.addAllocation(allocation);
    }
    residency.commit();
    let event = device.newSharedEvent().unwrap();
    let shared_event: &ProtocolObject<dyn MTLSharedEvent> = &event;
    let event_ref: &ProtocolObject<dyn MTLEvent> = shared_event.as_ref();

    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let mut allocators = Vec::with_capacity(STEPS);
    let mut command_buffers = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let allocator = device.newCommandAllocator().unwrap();
        let command_buffer = device.newCommandBuffer().unwrap();
        command_buffer.beginCommandBufferWithAllocator(&allocator);
        command_buffer.useResidencySet(&residency);
        let encoder = command_buffer.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipeline);
        encoder.setArgumentTable(Some(&table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
        encoder.endEncoding();
        command_buffer.endCommandBuffer();

        if step > 0 {
            queue.waitForEvent_value(event_ref, u64::try_from(step).unwrap());
        }
        let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = &command_buffer;
        let mut committed = [NonNull::from(command_buffer_ref)];
        // SAFETY: The pointer names one live command buffer and the count matches the array.
        unsafe {
            queue.commit_count(NonNull::from(&mut committed[0]), committed.len());
        }
        queue.signalEvent_value(event_ref, u64::try_from(step + 1).unwrap());
        allocators.push(allocator);
        command_buffers.push(command_buffer);
    }

    assert!(event.waitUntilSignaledValue_timeoutMS(u64::try_from(STEPS).unwrap(), 10_000));

    // SAFETY: Queue completion precedes these reads of the two shared u32 allocations.
    let (token, kv) = unsafe {
        (
            token.contents().cast::<u32>().read(),
            kv.contents().cast::<u32>().read(),
        )
    };
    assert_eq!(token, u32::try_from(STEPS).unwrap() + 1);
    assert_eq!(kv, u32::try_from(STEPS * (STEPS + 1) / 2).unwrap());
}
