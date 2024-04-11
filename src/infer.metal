#include <metal_stdlib>
#include <metal_compute>
#include <metal_simdgroup>
using namespace metal;

static constant int warpSize = 32;

inline float blockreduce_sum(threadgroup float* vs, float val, uint id) {
	val = simd_sum(val);

	vs[id / warpSize] = val;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	return simd_sum(vs[id % warpSize]);
}

inline float matmul_warppar(device float* x, device half* w, int i, int n, uint id) {
	int lane = id % warpSize;
	float val = 0.0f;
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		float2 ww = float2(*(device half2*)&w[i * n + j]);
		float2 xx = *(device float2*)&x[j];
		val += ww.x * xx.x;
		val += ww.y * xx.y;
	}
	return simd_sum(val);
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + exp(-x));
}

template <typename T>
kernel void kernel_embed(constant int& token_offset [[buffer(0)]], device float* o [[buffer(1)]], device T* weight [[buffer(2)]], uint id [[thread_position_in_grid]]) {
	o[id] = weight[id + token_offset];
}

template [[host_name("embed_half")]] kernel void kernel_embed<half>(constant int&, device float*, device half*, uint);

struct NormArgs {
	int size;
	float eps;
	bool ln;
};

[[host_name("rmsnorm")]] kernel void kernel_rmsnorm(constant NormArgs& args [[buffer(0)]], device float* o [[buffer(1)]], device float* x [[buffer(2)]], device float* weight [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int i = id;
	const int blockSize = 1024;
	int size = args.size;

	threadgroup float vs[32];

	float mean = 0.0f;
	if (args.ln) {
		// calculate sum (per thread)
		float sum = 0.0f;
		for (int j = i; j < size; j += blockSize) {
			sum += x[j];
		}

		// sum across threads in block
		mean = blockreduce_sum(vs, sum, id) / size;
	}

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j] - mean;
		ss += v * v;
	}

	// sum across threads in block
	ss = blockreduce_sum(vs, ss, id);

	float scale = rsqrt(ss / size + args.eps);

	for (int j = i; j < size; j += blockSize) {
		o[j] = (x[j] - mean) * weight[j] * scale;
	}
}

template <typename T, bool act_gelu>
kernel void kernel_ffn13(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w1 [[buffer(3)]], device T* w3 [[buffer(4)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float v1 = matmul_warppar(x, w1, j, n, id);
	float v3 = matmul_warppar(x, w3, j, n, id);

	if (id % warpSize == 0) {
		xout[j] = (act_gelu ? gelu(v1) : silu(v1)) * v3;
	}
}

template [[host_name("ffn13_silu_half")]] kernel void kernel_ffn13<half, false>(constant int&, device float*, device float*, device half*, device half*, uint);
template [[host_name("ffn13_gelu_half")]] kernel void kernel_ffn13<half, true>(constant int&, device float*, device float*, device half*, device half*, uint);

template <typename T>
kernel void kernel_ffn2(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w2 [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w2, j, n, id);

	if (id % warpSize == 0) {
		xout[j] += val;
	}
}

template [[host_name("ffn2_half")]] kernel void kernel_ffn2<half>(constant int&, device float*, device float*, device half*, uint);

template <typename T>
kernel void kernel_output(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w, j, n, id);

	if (id % warpSize == 0) {
		xout[j] = val;
	}
}

template [[host_name("output_half")]] kernel void kernel_output<half>(constant int&, device float*, device float*, device half*, uint);
