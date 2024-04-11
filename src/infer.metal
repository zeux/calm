#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void kernel_embed(constant int& token_offset [[buffer(0)]], device float* o [[buffer(1)]], device T* weight [[buffer(2)]], uint id [[thread_position_in_grid]]) {
	o[id] = weight[id + token_offset];
}

template [[host_name("embed_half")]] kernel void kernel_embed<half>(constant int&, device float*, device half*, uint);