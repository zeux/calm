#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void profiler_begin();
void profiler_trigger(const char* name, size_t bytes);
void profiler_endsync();

void profiler_reset();
void profiler_dump();

#ifdef __cplusplus
}
#endif
