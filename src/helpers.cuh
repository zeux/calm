#pragma once

#include <assert.h>
#include <cuda_fp16.h>

__device__ inline float warpreduce_sum(float v) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, mask);
	}
	return v;
}

__device__ inline float warpreduce_max(float v) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
	}
	return v;
}

// regular mat*vec; naive and unoptimized (won't reach peak bw or flops)
__device__ inline float matmul(float* x, half* w, int i, int n) {
	float val = 0.0f;
	for (int j = 0; j < n; j++) {
		val += float(w[i * n + j]) * x[j];
	}
	return val;
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
__device__ inline float matmul_warppar(float* x, half* w, int i, int n) {
	assert(n % warpSize == 0);
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	for (int j = lane; j < n; j += warpSize) {
		val += float(w[i * n + j]) * x[j];
	}
	return warpreduce_sum(val);
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for half weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar_half2(float* x, half* w, int i, int n) {
	assert(n % (warpSize * 2) == 0);
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		float2 ww = __half22float2(*(half2*)&w[i * n + j]);
		val += ww.x * x[j];
		val += ww.y * x[j + 1];
	}
	return warpreduce_sum(val);
}
