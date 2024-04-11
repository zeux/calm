#include <metal_stdlib>
using namespace metal;

kernel void kernel_basic(
    device float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant float& scale [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * scale;
}
