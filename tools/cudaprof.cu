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

#define BUFFER_SIZE (8 * 1024 * 1024)
#define MAX_TOKENS (1024 * 1024)
#define MAX_KERNELS 1024

struct KernelInfo {
	const char* name;

	float total_time;
	int calls;
	float call_avg;
	float call_m2;
	float peak_bw;
	float peak_util;
	int limit_occ;
};

static CUpti_ActivityDevice3 device;

static uint64_t tokens[MAX_TOKENS];

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
		case CUPTI_ACTIVITY_KIND_DEVICE: {
			device = *(CUpti_ActivityDevice3*)record;
			break;
		}

		case CUPTI_ACTIVITY_KIND_KERNEL:
		case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
			CUpti_ActivityKernel5* activity = (CUpti_ActivityKernel5*)record;
			KernelInfo* info = get_kernel(activity->name);

			float time = (float)(activity->end - activity->start) / 1e6;
			uint64_t token = tokens[activity->correlationId % MAX_TOKENS];

			info->total_time += time;

			// Welford's algorithm
			float delta = time - info->call_avg;
			info->calls++;
			info->call_avg += delta / info->calls;
			info->call_m2 += delta * (time - info->call_avg);

			// update peak bandwidth for kernel calls that specify profiling token as the first argument
			if ((token >> 48) == 0xCDAF) {
				uint64_t bytes = token & ((1ull << 48) - 1);
				float bw = ((double)bytes / 1e9) / (time / 1e3);
				info->peak_bw = fmaxf(info->peak_bw, bw);
			}

			int blocks = activity->gridX * activity->gridY * activity->gridZ;
			int blocks_rounded = (blocks + device.numMultiprocessors - 1) / device.numMultiprocessors * device.numMultiprocessors;
			info->peak_util = fmaxf(info->peak_util, (float)blocks / blocks_rounded);

			int block_size = activity->blockX * activity->blockY * activity->blockZ;
			int block_size_warps = (block_size + device.numThreadsPerWarp - 1) / device.numThreadsPerWarp;

			int occ_limit_blocks = device.maxBlocksPerMultiprocessor;
			int occ_limit_warps = device.maxWarpsPerMultiprocessor / block_size_warps;
			int occ_limit_smem = (activity->sharedMemoryExecuted) / (activity->staticSharedMemory + activity->dynamicSharedMemory + 1024);
			int occ_limit_regs = device.maxRegistersPerMultiprocessor / (((activity->registersPerThread * device.numThreadsPerWarp + 255) & ~255) * block_size_warps);
			int occ_limit = min(occ_limit_blocks, min(occ_limit_warps, min(occ_limit_smem, occ_limit_regs)));
			info->limit_occ = max(info->limit_occ, occ_limit * block_size_warps);
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

static void CUPTIAPI callback_handler(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
	const CUpti_CallbackData* cbinfo = (CUpti_CallbackData*)cbdata;

	switch (domain) {
	case CUPTI_CB_DOMAIN_RUNTIME_API: {
		switch (cbid) {
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: {
			if (cbinfo->callbackSite == CUPTI_API_ENTER) {
				cudaLaunchKernel_v7000_params* params = (cudaLaunchKernel_v7000_params*)cbinfo->functionParams;
				tokens[cbinfo->correlationId % MAX_TOKENS] = *(uint64_t*)*params->args;
			}
			break;
		}
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000: {
			if (cbinfo->callbackSite == CUPTI_API_ENTER) {
				cudaLaunchCooperativeKernel_v9000_params* params = (cudaLaunchCooperativeKernel_v9000_params*)cbinfo->functionParams;
				tokens[cbinfo->correlationId % MAX_TOKENS] = *(uint64_t*)*params->args;
			}
			break;
		}
		default:
			break;
		}
		break;
	}
	default:
		break;
	}
}

static void atexit_handler(void) {
	CUPTI_CHECK(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));

	if (n_kernels) {
		printf("\n");
		printf("%20s%10s%23s%12s%15s%25s\n", "Kernel", "Time", "Avg Time (us)", "Calls", "BW (GB/s)", "Utilization (limit)");
		printf("%20s%10s%23s%12s%15s%25s\n", "---", "---", "---", "---", "---", "---");

		float total_time = 0;
		for (int i = 0; i < n_kernels; i++) {
			total_time += kernels[i].total_time;
		}

		for (int i = 0; i < n_kernels; i++) {
			KernelInfo* kernel = &kernels[i];

			const char* name = kernel->name;
			size_t length = strlen(name);

			if (strncmp(name, "_Z", 2) == 0 && length >= 2) {
				name += 2;
				char* end;
				length = strtoul(name, &end, 10);
				name = end;
				length = length > strlen(name) ? strlen(name) : length;
			}

			if (strncmp(name, "kernel_", 7) == 0 && length >= 7) {
				name += 7;
				length -= 7;
			}

			const char* namecont = "";

			if (length > 20) {
				length = 19;
				namecont = "…";
			}

			char avgtime[64];
			snprintf(avgtime, sizeof(avgtime), "%.2f ± %.2f",
			         kernel->call_avg * 1e3,
			         sqrtf(kernel->call_m2 / kernel->calls) * 1e3);

			char util[64];
			snprintf(util, sizeof(util), "%.0f%% SMs, %d wrp/SM", kernel->peak_util * 100, kernel->limit_occ);

			printf("%20.*s%s%9.1f%%%24s%12d%15.1f%25s\n", (int)length, name, namecont,
			       kernel->total_time / total_time * 100, avgtime, kernel->calls, kernel->peak_bw, util);
		}
	}
}

extern "C" int InitializeInjection(void) {
	atexit(&atexit_handler);

	CUpti_SubscriberHandle subscriber;
	CUPTI_CHECK(cuptiSubscribe(&subscriber, callback_handler, NULL));
	CUPTI_CHECK(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
	CUPTI_CHECK(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000));

	const char* sync = getenv("PROF_SYNC");

	// note: KIND_KERNEL serializes kernel launches; KIND_CONCURRENT_KERNEL does not but it results in less stable timings
	if (sync && atoi(sync)) {
		CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
	} else {
		CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
	}

	CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));

	CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
	return 1;
}
