#include "model.h"

#include <Metal/Metal.h>

extern unsigned char infer_metallib[];
extern unsigned int infer_metallib_len;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> kernels[256];

static void dispatch(id<MTLComputeCommandEncoder> encoder, unsigned int thread_groups, unsigned int thread_group_size, const char* name, void* params, size_t params_size, void** buffers, size_t buffer_count) {
    id<MTLComputePipelineState> state = nil;
    for (size_t i = 0; kernels[i]; ++i) {
        if (strcmp(kernels[i].label.UTF8String, name) == 0) {
            state = kernels[i];
            break;
        }
    }
    assert(state);

    [encoder setComputePipelineState:state];
    [encoder setBytes:params length:params_size atIndex:0];
    for (size_t i = 0; i < buffer_count; ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i+1];
    }

    [encoder dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
}

void init_metal(void) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    assert(devices.count > 0);

    device = devices[0];
    queue = [device newCommandQueue];

    dispatch_data_t lib_data = dispatch_data_create(infer_metallib, infer_metallib_len, dispatch_get_main_queue(), ^{});

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
    assert(library);

    NSArray<NSString*>* functions = library.functionNames;
    for (size_t i = 0; i < functions.count; i++) {
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor alloc];
        descriptor.computeFunction = [library newFunctionWithName:functions[i]];
        descriptor.label = functions[i];

        id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
        assert(computePipelineState);
        kernels[i] = computePipelineState;
    }
}

void* upload_metal(void* host, size_t size) {
    assert(device);
    id<MTLBuffer> buffer = [device newBufferWithBytes:host length:size options:MTLResourceStorageModeShared];
    return buffer;
}

void prepare_metal(struct Transformer* transformer) {
    assert(device);
    printf("# Metal: %s\n", device.name.UTF8String);

    // Create buffers for input and output data
    float inputData[] = {1.0, 2.0, 3.0, 4.0};
    float outputData[4] = {0};
    float scale = 0.5;
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputData length:sizeof(inputData) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithBytes:outputData length:sizeof(outputData) options:MTLResourceStorageModeShared];
    
    // Create a command buffer
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    
    // Create a compute command encoder
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    dispatch(computeEncoder, 1, 4, "kernel_basic", &scale, sizeof(scale), (void*[]){ inputBuffer, outputBuffer }, 2);
    
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