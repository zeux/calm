#include "model.h"

#include <Metal/Metal.h>

extern unsigned char infer_metallib[];
extern unsigned int infer_metallib_len;

static id<MTLDevice> device;

void init_metal(void) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    assert(devices.count > 0);

    device = devices[0];
}

void* upload_metal(void* host, size_t size) {
    assert(device);
    id<MTLBuffer> buffer = [device newBufferWithBytes:host length:size options:MTLResourceStorageModeShared];
    return buffer;
}

void prepare_metal(struct Transformer* transformer) {
    assert(device);
    printf("# Metal: %s\n", device.name.UTF8String);

    dispatch_data_t lib_data = dispatch_data_create(infer_metallib, infer_metallib_len, dispatch_get_main_queue(), ^{});
    
    // Load the kernel function from default library
    NSError *error = nil;
    id<MTLLibrary> defaultLibrary = [device newLibraryWithData:lib_data error:&error];
    id<MTLFunction> kernelFunction = [defaultLibrary newFunctionWithName:@"kernel_basic"];
    
    // Create a compute pipeline state
    id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!computePipelineState) {
        NSLog(@"Failed to create compute pipeline state: %@", error);
        return;
    }
    
    // Create a command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create buffers for input and output data
    float inputData[] = {1.0, 2.0, 3.0, 4.0};
    float outputData[4] = {0};
    float scale = 0.5;
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputData length:sizeof(inputData) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithBytes:outputData length:sizeof(outputData) options:MTLResourceStorageModeShared];
    id<MTLBuffer> scaleBuffer = [device newBufferWithBytes:&scale length:sizeof(scale) options:MTLResourceStorageModeShared];
    
    // Create a command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    // Create a compute command encoder
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:computePipelineState];
    [computeEncoder setBuffer:inputBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:outputBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:scaleBuffer offset:0 atIndex:2];
    
    // Dispatch the compute kernel
    MTLSize gridSize = MTLSizeMake(4, 1, 1); // Process the 4 elements
    NSUInteger threadGroupSize = computePipelineState.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > gridSize.width) {
        threadGroupSize = gridSize.width;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    
    // End encoding and commit the command buffer
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Read back the data
    memcpy(outputData, outputBuffer.contents, sizeof(outputData));
    for (int i = 0; i < 4; i++) {
        NSLog(@"Output %d: %f", i, outputData[i]);
    }
}

float* forward_metal(struct Transformer* transformer, int token, int pos, unsigned flags) {
    return NULL;
}