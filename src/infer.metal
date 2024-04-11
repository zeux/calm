#include <metal_stdlib>
using namespace metal;

kernel void kernel_basic(
    constant float& scale [[buffer(0)]],
    device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    output[id] = input[id] * scale;
}
