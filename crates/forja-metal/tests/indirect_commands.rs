//! Hardware semantics for Metal indirect compute commands.

#![cfg(target_os = "macos")]

use std::{ffi::c_void, ptr::NonNull};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
    MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLAllocation, MTLBuffer,
    MTLCompileOptions, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLEvent, MTLFunctionConstantValues,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLLibrary, MTLPipelineOption, MTLResidencySet,
    MTLResidencySetDescriptor, MTLResourceOptions, MTLSharedEvent, MTLSize, MTLStages,
};

const SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

constant uint seed_adjustment [[function_constant(0)]];

kernel void seed(
    device const uint *base [[buffer(0)]],
    device uint *value [[buffer(1)]],
    constant uint &left [[buffer(2)]],
    constant uint &right [[buffer(3)]],
    threadgroup uint *scratch [[threadgroup(0)]],
    uint lane [[thread_position_in_threadgroup]]) {
    if (lane == 0) {
        scratch[0] = 40;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        value[0] = scratch[0] + base[0] + left + right + seed_adjustment;
    }
}

kernel void increment(device uint *value [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        value[0] += 1;
    }
}
";

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn indirect_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Pipeline {
    let function = if name == "seed" {
        let values = MTLFunctionConstantValues::new();
        let adjustment = 1_u32;
        // SAFETY: The value is a live u32 and index zero declares a uint function constant.
        unsafe {
            values.setConstantValue_type_atIndex(
                NonNull::from(&adjustment).cast::<c_void>(),
                MTLDataType::UInt,
                0,
            );
        }
        library
            .newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
            .unwrap()
    } else {
        library
            .newFunctionWithName(&NSString::from_str(name))
            .unwrap()
    };
    let descriptor = MTLComputePipelineDescriptor::new();
    descriptor.setComputeFunction(Some(&function));
    descriptor.setSupportIndirectCommandBuffers(true);
    device
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
        .unwrap()
}

fn encode_indirect(
    command: &ProtocolObject<dyn MTLIndirectComputeCommand>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    threadgroups: MTLSize,
    threads: MTLSize,
) {
    command.setComputePipelineState(pipeline);
    // SAFETY: The buffer is retained until GPU completion and index zero is declared by both
    // kernels and covered by the indirect command buffer descriptor.
    unsafe {
        command.setKernelBuffer_offset_atIndex(buffer, 0, 0);
    }
    command.concurrentDispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
}

fn barrier(encoder: &ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn indirect_compute_preserves_dependencies_and_bindings() {
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let queue = device.newMTL4CommandQueue().unwrap();
    let allocator = device.newCommandAllocator().unwrap();
    let command_buffer = device.newCommandBuffer().unwrap();
    let library = device
        .newLibraryWithSource_options_error(
            &NSString::from_str(SOURCE),
            Some(&MTLCompileOptions::new()),
        )
        .unwrap();
    let seed = indirect_pipeline(&device, &library, "seed");
    let increment = indirect_pipeline(&device, &library, "increment");
    let buffer = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let base = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let left = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let right = device
        .newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    // SAFETY: Each shared allocation contains one u32 and GPU work has not started.
    unsafe {
        base.contents().cast::<u32>().write(1);
        left.contents().cast::<u32>().write(2);
        right.contents().cast::<u32>().write(3);
    }

    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    descriptor.setInheritPipelineState(false);
    descriptor.setInheritBuffers(false);
    descriptor.setMaxKernelBufferBindCount(4);
    // SAFETY: Threadgroup memory index zero is the only index used below.
    unsafe {
        descriptor.setMaxKernelThreadgroupMemoryBindCount(1);
    }
    // SAFETY: The descriptor permits all commands and bindings encoded at the three valid indices.
    let indirect = unsafe {
        device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &descriptor,
            3,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .unwrap();
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    let thirty_two = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    // SAFETY: All command indices are below the indirect buffer's command count.
    let first = unsafe { indirect.indirectComputeCommandAtIndex(0) };
    encode_indirect(&first, &seed, &base, one, thirty_two);
    // SAFETY: The descriptor covers all indices and every buffer is retained through completion.
    unsafe {
        first.setKernelBuffer_offset_atIndex(&buffer, 0, 1);
        first.setKernelBuffer_offset_atIndex(&left, 0, 2);
        first.setKernelBuffer_offset_atIndex(&right, 0, 3);
    }
    // SAFETY: The descriptor reserves threadgroup memory index zero and the kernel uses four bytes.
    unsafe {
        first.setThreadgroupMemoryLength_atIndex(4, 0);
    }
    first.setBarrier();
    // SAFETY: All command indices are below the indirect buffer's command count.
    let second = unsafe { indirect.indirectComputeCommandAtIndex(1) };
    encode_indirect(&second, &increment, &buffer, one, one);
    second.setBarrier();
    // SAFETY: All command indices are below the indirect buffer's command count.
    let third = unsafe { indirect.indirectComputeCommandAtIndex(2) };
    encode_indirect(&third, &increment, &buffer, one, one);

    let table_descriptor = MTL4ArgumentTableDescriptor::new();
    table_descriptor.setMaxBufferBindCount(1);
    let table = device
        .newArgumentTableWithDescriptor_error(&table_descriptor)
        .unwrap();
    // SAFETY: Index zero is within the table and the buffer remains live until completion.
    unsafe {
        table.setAddress_atIndex(buffer.gpuAddress(), 0);
    }
    let residency = device
        .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
        .unwrap();
    let allocation: &ProtocolObject<dyn MTLAllocation> = buffer.as_ref();
    residency.addAllocation(allocation);
    for raw in [&base, &left, &right] {
        let allocation: &ProtocolObject<dyn MTLAllocation> = raw.as_ref();
        residency.addAllocation(allocation);
    }
    let allocation: &ProtocolObject<dyn MTLAllocation> = indirect.as_ref();
    residency.addAllocation(allocation);
    residency.commit();

    command_buffer.beginCommandBufferWithAllocator(&allocator);
    command_buffer.useResidencySet(&residency);
    let encoder = command_buffer.computeCommandEncoder().unwrap();
    // SAFETY: The range covers the first two initialized commands.
    unsafe {
        encoder.executeCommandsInBuffer_withRange(&indirect, NSRange::new(0, 2));
    }
    barrier(&encoder);
    encoder.setComputePipelineState(&increment);
    encoder.setArgumentTable(Some(&table));
    encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
    barrier(&encoder);
    // SAFETY: The range covers the initialized third command.
    unsafe {
        encoder.executeCommandsInBuffer_withRange(&indirect, NSRange::new(2, 1));
    }
    encoder.endEncoding();
    command_buffer.endCommandBuffer();

    let command_buffer_ref: &ProtocolObject<dyn MTL4CommandBuffer> = &command_buffer;
    let mut command_buffers = [NonNull::from(command_buffer_ref)];
    // SAFETY: The pointer refers to one live command buffer and the count matches the array.
    unsafe {
        queue.commit_count(
            NonNull::from(&mut command_buffers[0]),
            command_buffers.len(),
        );
    }
    let event = device.newSharedEvent().unwrap();
    let event_ref: &ProtocolObject<dyn MTLSharedEvent> = &event;
    let event_ref: &ProtocolObject<dyn MTLEvent> = event_ref.as_ref();
    queue.signalEvent_value(event_ref, 1);
    assert!(event.waitUntilSignaledValue_timeoutMS(1, 10_000));

    // SAFETY: Shared storage is four bytes long and GPU completion precedes this read.
    let result = unsafe { buffer.contents().cast::<u32>().read() };
    assert_eq!(result, 50);
}
