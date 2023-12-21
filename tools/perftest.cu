#include <time.h>
#include <stdio.h>

#include "../src/helpers.cuh"

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUDA_SYNC() CUDA_CHECK(cudaDeviceSynchronize())

static void* cuda_devicealloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, size));
	return ptr;
}

__global__ static void kernel_matmul_perf(float* xout, float* x, half* w, int n, int d) {
	int i = blockIdx.x;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n);

	if (threadIdx.x == 0) {
		xout[i] = val;
	}
}

static long time_in_ms() {
	// return time in milliseconds, for benchmarking the model speed
	struct timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main() {
	// benchmark memcpy performance
	for (int dim = 16384; dim <= 32768; dim += 4096) {
		half* w = (half*)cuda_devicealloc(dim * dim * sizeof(half));
		CUDA_CHECK(cudaMemset(w, 0, dim * dim * sizeof(half)));

		half* wc = (half*)cuda_devicealloc(dim * dim * sizeof(half));

		// warmup
		CUDA_CHECK(cudaMemcpy(wc, w, dim * dim * sizeof(half), cudaMemcpyDeviceToDevice));
		CUDA_SYNC();

		int n = 10000000 / dim;

		// benchmark
		long start = time_in_ms();

		for (int i = 0; i < n; i++) {
			CUDA_CHECK(cudaMemcpy(wc, w, dim * dim * sizeof(half), cudaMemcpyDeviceToDevice));
		}

		CUDA_SYNC();

		long end = time_in_ms();

		printf("memcpy %d: %.2f ms/op, %.2f GB/s\n", dim, double(end - start) / n, (2 * double(dim) * dim * sizeof(half) * n / 1e9) / (double(end - start) / 1e3));

		CUDA_CHECK(cudaFree(wc));
		CUDA_CHECK(cudaFree(w));
	}

	printf("\n");

	// benchmark matmul performance
	for (int dim = 16384; dim <= 32768; dim += 4096) {
		half* w = (half*)cuda_devicealloc(dim * dim * sizeof(half));
		CUDA_CHECK(cudaMemset(w, 0, dim * dim * sizeof(half)));

		float* x = (float*)cuda_devicealloc(dim * sizeof(float));
		float* xout = (float*)cuda_devicealloc(dim * sizeof(float));

		int n = 10000000 / dim;

		// warmup
		kernel_matmul_perf<<<dim, 32>>>(xout, x, w, dim, dim);
		CUDA_SYNC();

		// benchmark
		long start = time_in_ms();

		for (int i = 0; i < n; i++) {
			kernel_matmul_perf<<<dim, 32>>>(xout, x, w, dim, dim);
		}

		CUDA_SYNC();

		long end = time_in_ms();

		printf("matmul %d: %.2f ms/op, %.2f GB/s\n", dim, double(end - start) / n, (double(dim) * dim * sizeof(half) * n / 1e9) / (double(end - start) / 1e3));

		CUDA_CHECK(cudaFree(xout));
		CUDA_CHECK(cudaFree(x));
		CUDA_CHECK(cudaFree(w));
	}
}
