// Based on NVIDIA's cupti_trace_injection sample
#include <assert.h>
#include <stdio.h>

#include <cuda.h>
#include <cupti.h>

#define CUPTI_CHECK(call)                                            \
	do {                                                             \
		CUptiResult _status = call;                                  \
		if (_status != CUPTI_SUCCESS) {                              \
			const char* err = "?";                                   \
			cuptiGetResultString(_status, &err);                     \
			fprintf(stderr, "CUPTI error in %s at %s:%d: %s (%d)\n", \
			        __FUNCTION__, __FILE__, __LINE__, err, _status); \
			abort();                                                 \
		}                                                            \
	} while (0)

#define BUFFER_SIZE 8 * 1024 * 1024
#define MAX_KERNELS 1024

struct KernelInfo {
	const char* name;

	float total_time;
	float min_time;
	float peak_bw;
	int calls;
};

static KernelInfo kernels[MAX_KERNELS];
static int n_kernels;

static KernelInfo* get_kernel(const char* name) {
	for (int i = 0; i < n_kernels; i++) {
		if (strcmp(kernels[i].name, name) == 0) {
			return &kernels[i];
		}
	}

	assert(n_kernels < MAX_KERNELS);
	KernelInfo* kernel = &kernels[n_kernels++];
	kernel->name = name;

	return kernel;
}

static void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
	*size = BUFFER_SIZE;
	*buffer = (uint8_t*)malloc(BUFFER_SIZE);
	*maxNumRecords = 0;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId, uint8_t* buffer, size_t size, size_t validSize) {
	CUpti_Activity* record = NULL;

	for (;;) {
		CUptiResult status = cuptiActivityGetNextRecord(buffer, validSize, &record);
		if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
			break;
		}
		CUPTI_CHECK(status);

		switch (record->kind) {
		case CUPTI_ACTIVITY_KIND_KERNEL:
		case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
			CUpti_ActivityKernel8* activity = (CUpti_ActivityKernel8*)record;
			KernelInfo* info = get_kernel(activity->name);

			float time = (float)(activity->end - activity->start) / 1e6;

			info->calls++;
			info->min_time = info->calls == 1 ? time : fminf(info->min_time, time);
			info->total_time += time;
			break;
		}
		default:
			break;
		}
	}

	free(buffer);

	size_t dropped = 0;
	CUPTI_CHECK(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));

	if (dropped != 0) {
		printf("WARNING: dropped %u CUPTI activity records.\n", (unsigned int)dropped);
	}
}

static void atexit_handler(void) {
	CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

	if (n_kernels) {
		printf("\n");
		printf("%20s%20s%20s%20s%20s%20s\n", "Kernel", "Total Time (%)", "Calls", "Avg Time (us)", "Min Time (us)", "Peak BW (GB/s)");
		printf("%20s%20s%20s%20s%20s%20s\n", "---", "---", "---", "---", "---", "---");

		float total_time = 0;
		for (int i = 0; i < n_kernels; i++) {
			total_time += kernels[i].total_time;
		}

		for (int i = 0; i < n_kernels; i++) {
			KernelInfo* kernel = &kernels[i];

			const char* name = kernel->name;
			size_t length = strlen(name);

			if (strncmp(name, "_Z", 2) == 0) {
				name += 2;
				char* end;
				length = strtoul(name, &end, 10);
				name = end;
			}

			if (strncmp(name, "kernel_", 7) == 0) {
				name += 7;
				length -= 7;
			}

			printf("%20.*s%19.1f%%%20d%20.1f%20.1f%20.2f\n", (int)length, name, kernel->total_time / total_time * 100, kernel->calls, kernel->total_time / kernel->calls * 1e3, kernel->min_time * 1e3, kernel->peak_bw);
		}
	}
}

extern "C" int InitializeInjection(void) {
	atexit(&atexit_handler);

	// note: KIND_KERNEL serializes kernel launches; KIND_CONCURRENT_KERNEL does not but it results in less stable timings
	CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

	CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
	return 1;
}

int main() {

}
