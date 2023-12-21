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

static float events_mintime(cudaEvent_t* events, size_t nevents) {
	float mint = FLT_MAX;

	for (size_t ei = 1; ei < nevents; ei++) {
		float t;
		CUDA_CHECK(cudaEventElapsedTime(&t, events[ei - 1], events[ei]));
		mint = mint < t ? mint : t;
	}

	return mint;
}

int main() {
	cudaEvent_t events[1000];

	for (size_t ei = 0; ei < sizeof(events) / sizeof(events[0]); ei++) {
		CUDA_CHECK(cudaEventCreate(&events[ei]));
	}

	// benchmark memcpy performance
	for (size_t mb = 512; mb <= 2048; mb *= 2) {
		void* w = cuda_devicealloc(mb * 1024 * 1024);
		CUDA_CHECK(cudaMemset(w, 0, mb * 1024 * 1024));

		void* wc = cuda_devicealloc(mb * 1024 * 1024);

		CUDA_CHECK(cudaEventRecord(events[0]));

		for (size_t ei = 1; ei < sizeof(events) / sizeof(events[0]); ei++) {
			CUDA_CHECK(cudaMemcpy(wc, w, mb * 1024 * 1024, cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaEventRecord(events[ei]));
		}

		CUDA_CHECK(cudaEventSynchronize(events[sizeof(events) / sizeof(events[0]) - 1]));

		float mint = events_mintime(events, sizeof(events) / sizeof(events[0]));

		printf("memcpy %d MB: %.2f ms peak, %.2f GB/s\n", (int)mb, mint, (2 * double(mb) * 1024 * 1024 / 1e9) / (mint / 1e3));

		CUDA_CHECK(cudaFree(wc));
		CUDA_CHECK(cudaFree(w));
	}

	printf("\n");

	// benchmark matmul performance
	const int dims[][2] = {
	    // ~2 GB matrix that definitely isn't impacted by cache effects
	    {32768, 32768},
	    // generic sizes
	    {4096, 4096},
	    {8192, 8192},
	    {16384, 16384},
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

	for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
		if (di == 1) {
			printf("\n");
		}

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

		float mint = events_mintime(events, sizeof(events) / sizeof(events[0]));

		printf("matmul %dx%d: %.2f ms peak, %.2f GB/s\n", n, d, mint, (double(n) * d * sizeof(half) / 1e9) / (mint / 1e3));

		CUDA_CHECK(cudaFree(xout));
		CUDA_CHECK(cudaFree(x));
		CUDA_CHECK(cudaFree(w));
	}

	for (size_t ei = 0; ei < sizeof(events) / sizeof(events[0]); ei++) {
		CUDA_CHECK(cudaEventDestroy(events[ei]));
	}
}
