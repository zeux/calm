#include <float.h>
#include <stdio.h>
#include <time.h>

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

__global__ static void kernel_matmul(float* xout, float* x, half* w, int n, int d) {
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
	const int dims[][2] = {
	    // generic large sizes
	    {4096, 4096},
	    {8192, 8192},
	    {16384, 16384},
	    {32768, 32768},
	    // mistral 7b
	    {14336, 4096},
	    {4096, 1024},
	    {4096, 14336},
	    {4096, 32000},
	    // llama2 7b
	    {11008, 4096},
	    {4096, 11008},
	    {4096, 32000},
	};

	cudaEvent_t events[1000];

	for (size_t ei = 0; ei < sizeof(events) / sizeof(events[0]); ei++) {
		CUDA_CHECK(cudaEventCreate(&events[ei]));
	}

	for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
		int n = dims[di][0], d = dims[di][1];

		half* w = (half*)cuda_devicealloc(n * d * sizeof(half));
		CUDA_CHECK(cudaMemset(w, 0, n * d * sizeof(half)));

		float* x = (float*)cuda_devicealloc(n * sizeof(float));
		float* xout = (float*)cuda_devicealloc(d * sizeof(float));

		CUDA_CHECK(cudaEventRecord(events[0]));

		for (size_t ei = 1; ei < sizeof(events) / sizeof(events[0]); ei++) {
			kernel_matmul<<<d, 32>>>(xout, x, w, n, d);
			CUDA_CHECK(cudaEventRecord(events[ei]));
		}

		CUDA_CHECK(cudaEventSynchronize(events[sizeof(events) / sizeof(events[0]) - 1]));

		float mint = FLT_MAX;

		for (size_t ei = 1; ei < sizeof(events) / sizeof(events[0]); ei++) {
			float t;
			CUDA_CHECK(cudaEventElapsedTime(&t, events[ei - 1], events[ei]));
			mint = mint < t ? mint : t;
		}

		printf("matmul %dx%d: %.2f ms peak, %.2f GB/s\n", n, d, mint, (double(n) * d * sizeof(half) / 1e9) / (mint / 1e3));

		CUDA_CHECK(cudaFree(xout));
		CUDA_CHECK(cudaFree(x));
		CUDA_CHECK(cudaFree(w));
	}
}
