#include "profiler.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#define MAX_KERNELS 100
#define MAX_EVENTS 10000

struct ProfilerKernel {
	const char* name;

	float total_time;
	float min_time;
	float peak_bw;
	int calls;
};

struct ProfilerTrigger {
	cudaEvent_t event;

	const char* name;
	size_t bytes;
};

struct Profiler {
	cudaEvent_t start;

	struct ProfilerTrigger triggers[MAX_EVENTS];
	int n_triggers;

	struct ProfilerKernel kernels[MAX_KERNELS];
	int n_kernels;
} profiler;

static int profiler_enabled = -1;

static struct ProfilerKernel* get_kernel(const char* name) {
	for (int i = 0; i < profiler.n_kernels; i++) {
		if (strcmp(profiler.kernels[i].name, name) == 0) {
			return &profiler.kernels[i];
		}
	}

	assert(profiler.n_kernels < MAX_KERNELS);
	struct ProfilerKernel* kernel = &profiler.kernels[profiler.n_kernels++];
	kernel->name = name;

	return kernel;
}

void profiler_begin() {
	if (profiler_enabled < 0) {
		const char* env = getenv("CALM_PROF");
		profiler_enabled = env && atoi(env);

		if (profiler_enabled) {
			cudaEventCreate(&profiler.start);

			for (int i = 0; i < MAX_EVENTS; i++) {
				cudaEventCreate(&profiler.triggers[i].event);
			}
		}
	}

	if (profiler_enabled <= 0)
		return;

	cudaEventRecord(profiler.start, 0);
}

void profiler_trigger(const char* name, size_t bytes) {
	if (profiler_enabled <= 0)
		return;

	assert(profiler.n_triggers < MAX_EVENTS);
	struct ProfilerTrigger* trigger = &profiler.triggers[profiler.n_triggers++];

	cudaEventRecord(trigger->event, 0);
	trigger->name = name;
	trigger->bytes = bytes;
}

void profiler_endsync() {
	if (profiler.n_triggers == 0)
		return;

	cudaEventSynchronize(profiler.triggers[profiler.n_triggers - 1].event);

	cudaEvent_t last_event = profiler.start;

	for (int i = 0; i < profiler.n_triggers; i++) {
		struct ProfilerTrigger* trigger = &profiler.triggers[i];

		float ms;
		cudaEventElapsedTime(&ms, last_event, trigger->event);

		float bw = ((double)trigger->bytes / 1e9) / (ms / 1e3);

		struct ProfilerKernel* kernel = get_kernel(trigger->name);

		kernel->calls++;
		kernel->total_time += ms;
		kernel->min_time = kernel->calls == 1 ? ms : fminf(kernel->min_time, ms);
		kernel->peak_bw = fmaxf(kernel->peak_bw, bw);

		last_event = trigger->event;
	}

	profiler.n_triggers = 0;

	char* env = getenv("CALM_PROFLOG");
	if (env && atoi(env)) {
		float total_ms;
		cudaEventElapsedTime(&total_ms, profiler.start, last_event);

		static int run = 0;
		fprintf(stderr, "%d\t%.2f\n", run++, total_ms);
	}
}

void profiler_reset() {
	profiler.n_kernels = 0;
	memset(profiler.kernels, 0, sizeof(profiler.kernels));
}

void profiler_dump() {
	if (profiler.n_kernels == 0)
		return;

	printf("\n");
	printf("%20s%20s%20s%20s%20s%20s\n", "Kernel", "Total Time (%)", "Calls", "Avg Time (us)", "Min Time (us)", "Peak BW (GB/s)");
	printf("%20s%20s%20s%20s%20s%20s\n", "---", "---", "---", "---", "---", "---");

	float total_time = 0;
	for (int i = 0; i < profiler.n_kernels; i++) {
		total_time += profiler.kernels[i].total_time;
	}

	for (int i = 0; i < profiler.n_kernels; i++) {
		struct ProfilerKernel* kernel = &profiler.kernels[i];

		printf("%20s%19.1f%%%20d%20.1f%20.1f%20.2f\n", kernel->name, kernel->total_time / total_time * 100, kernel->calls, kernel->total_time / kernel->calls * 1e3, kernel->min_time * 1e3, kernel->peak_bw);
	}
}
