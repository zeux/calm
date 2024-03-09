#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <stdint.h>

// note: we expect loads to be broken into units of up to 16b due to specified alignment
template <typename T, int N>
union _ALIGNAS(sizeof(T) * N) ablock {
	T v[N];
};

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

__device__ inline int warpreduce_maxi(int v) {
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

// gf4 decoding: 8 3-bit values + 1 fp8 scale are packed in a 32-bit word
__device__ inline float gf4_ff(uint32_t v, int k) {
	float s = fp8_e5m2_ff(v & 0xff) / -4.f; // we expect compiler to reuse this across multiple calls
	return (int((v >> (8 + k * 3)) & 7) - 4) * s;
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
__device__ inline float matmul_warppar(half* x, half* w, int i, int n) {
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		float2 ww = __half22float2(*(half2*)&w[i * n + j]);
		half2 xx = *(half2*)&x[j];
		val += ww.x * float(xx.x);
		val += ww.y * float(xx.y);
	}
	return warpreduce_sum(val);
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for fp8 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(half* x, __nv_fp8_e5m2* w, int i, int n) {
	int lane = threadIdx.x % warpSize;
	half2 val = {0, 0};
	for (int j = lane * 8; j < n; j += warpSize * 8) {
		ablock<__nv_fp8x2_e5m2, 4> wwp = *(ablock<__nv_fp8x2_e5m2, 4>*)&w[i * n + j];
		ablock<__half2_raw, 4> xxp = *(ablock<__half2_raw, 4>*)&x[j];

		val = __hfma2(half2(wwp.v[0]), half2(xxp.v[0]), val);
		val = __hfma2(half2(wwp.v[1]), half2(xxp.v[1]), val);
		val = __hfma2(half2(wwp.v[2]), half2(xxp.v[2]), val);
		val = __hfma2(half2(wwp.v[3]), half2(xxp.v[3]), val);
	}
	return warpreduce_sum(float(val.x + val.y));
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for gf4 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(half* x, uint32_t* w, int i, int n) {
	int lane = threadIdx.x % warpSize;
	if (n % (warpSize * 16) == 0) {
		float val = 0.0f;
		for (int j = lane * 8; j < n; j += warpSize * 16) {
			uint32_t wg0 = w[i * n / 8 + j / 8];
			uint32_t wg1 = w[i * n / 8 + j / 8 + warpSize];

			ablock<half, 8> xx0 = *(ablock<half, 8>*)&x[j];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += gf4_ff(wg0, k) * float(xx0.v[k]);
			}

			ablock<half, 8> xx1 = *(ablock<half, 8>*)&x[j + warpSize * 8];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += gf4_ff(wg1, k) * float(xx1.v[k]);
			}
		}
		return warpreduce_sum(val);
	} else {
		float val = 0.0f;
		for (int j = lane * 8; j < n; j += warpSize * 8) {
			uint32_t wg = w[i * n / 8 + j / 8];

			ablock<half, 8> xx = *(ablock<half, 8>*)&x[j];
#pragma unroll
			for (int k = 0; k < 8; ++k) {
				val += gf4_ff(wg, k) * float(xx.v[k]);
			}
		}
		return warpreduce_sum(val);
	}
}
