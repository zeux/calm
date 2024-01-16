#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>

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

__device__ inline float blocktranspose(float v, float def) {
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	__shared__ float sm[32];
	sm[warp] = v;
	__syncthreads();

	return lane < blockDim.x / warpSize ? sm[lane] : def;
}

__device__ inline float blockreduce_sum(float v) {
	v = warpreduce_sum(v);
	v = blocktranspose(v, 0.f);
	v = warpreduce_sum(v);
	return v;
}

__device__ inline float blockreduce_max(float v) {
	v = warpreduce_max(v);
	v = blocktranspose(v, -FLT_MAX);
	v = warpreduce_max(v);
	return v;
}

// fast fp8x4 => float4 conversion; drops unnecessary NaN handling from __nv_cvt_fp8_to_halfraw
__device__ inline float4 fp8x4_e5m2_ff(__nv_fp8x4_e5m2 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	return float4(v);
#else
	unsigned int vlo = v.__x, vhi = v.__x >> 16;
	__half2_raw hlo = {(unsigned short)(vlo << 8), (unsigned short)(vlo & 0xff00)};
	__half2_raw hhi = {(unsigned short)(vhi << 8), (unsigned short)(vhi & 0xff00)};
	float2 rlo = __internal_halfraw2_to_float2(hlo);
	float2 rhi = __internal_halfraw2_to_float2(hhi);
	float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
	return res;
#endif
}

__device__ inline float fp8_e5m2_ff(uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __half_raw h = __nv_cvt_fp8_to_halfraw(v, __NV_E5M2);
#else
    __half_raw h = {(unsigned short)(v << 8)};
#endif
    return __internal_halfraw_to_float(h);
}

// regular mat*vec; naive and unoptimized (won't reach peak bw or flops)
template <typename T>
__device__ inline float matmul(float* x, T* w, int i, int n) {
	float val = 0.0f;
	for (int j = 0; j < n; j++) {
		val += float(w[i * n + j]) * x[j];
	}
	return val;
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for half weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, half* w, int i, int n, int stride) {
	assert(n % (warpSize * 2) == 0);
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		float2 ww = __half22float2(*(half2*)&w[i * stride + j]);
		float2 xx = *(float2*)&x[j];
		val += ww.x * xx.x;
		val += ww.y * xx.y;
	}
	return warpreduce_sum(val);
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for fp8 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, __nv_fp8_e5m2* w, int i, int n, int stride) {
	assert(n % (warpSize * 4) == 0);
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	for (int j = lane * 4; j < n; j += warpSize * 4) {
		float4 ww = fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)&w[i * stride + j]);
		float4 xx = *(float4*)&x[j];
		val += ww.x * xx.x;
		val += ww.y * xx.y;
		val += ww.z * xx.z;
		val += ww.w * xx.w;
	}
	return warpreduce_sum(val);
}

union float8 {
	float4 g[2];
	float v[8];
};

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for gf4 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, uint32_t* w, int i, int n, int stride) {
	assert(n % (warpSize * 8) == 0);
	int lane = threadIdx.x % warpSize;
	if (n % (warpSize * 16) == 0) {
		float val = 0.0f;
		for (int j = lane * 8; j < n; j += warpSize * 16) {
			uint32_t wg0 = w[i * stride / 8 + j / 8];
			uint32_t wg1 = w[i * stride / 8 + j / 8 + warpSize];

			float wgs0 = -fp8_e5m2_ff(wg0 & 0xff) / 4.f;
			float wgs1 = -fp8_e5m2_ff(wg1 & 0xff) / 4.f;

			float8 xx0 = *(float8*)&x[j];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += (int((wg0 >> (8 + k * 3)) & 7) - 4) * wgs0 * xx0.v[k];
			}

			float8 xx1 = *(float8*)&x[j + warpSize * 8];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += (int((wg1 >> (8 + k * 3)) & 7) - 4) * wgs1 * xx1.v[k];
			}
		}
		return warpreduce_sum(val);
	} else {
		float val = 0.0f;
		for (int j = lane * 8; j < n; j += warpSize * 8) {
			uint32_t wg = w[i * stride / 8 + j / 8];
			float wgs = -fp8_e5m2_ff(wg & 0xff) / 4.f;

			float8 xx = *(float8*)&x[j];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += (int((wg >> (8 + k * 3)) & 7) - 4) * wgs * xx.v[k];
			}
		}
		return warpreduce_sum(val);
	}
}
