MAKEFLAGS+=-r -j

NVCC?=nvcc

BUILD=build

SOURCES=$(wildcard src/*.c)
SOURCES+=$(wildcard src/*.cu)
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

BINARY=$(BUILD)/run

CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math -fopenmp
LDFLAGS=-lm -fopenmp

CFLAGS+=-mf16c -mavx2

CUFLAGS+=-g -O2 -arch compute_80
LDFLAGS+=-lcudart

ifneq (,$(wildcard /usr/local/cuda))
    CFLAGS+=-I/usr/local/cuda/include
    LDFLAGS+=-L/usr/local/cuda/lib64
endif

all: $(BINARY)

format:
	clang-format -i src/* tools/*.cu

$(BUILD)/fuzz-tensors: src/tensors.c
	clang $(CFLAGS) -DFUZZING -O1 -fsanitize=address,fuzzer -o $@ $^

$(BUILD)/cudabench: tools/cudabench.cu
	$(NVCC) $< $(CUFLAGS) -MMD -MP -o $@

$(BUILD)/cudaprof: tools/cudaprof.cu
	$(NVCC) $< $(CUFLAGS) -Xcompiler -fPIC -shared -lcupti -MMD -MP -o $@

$(BINARY): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(BUILD)/cudabench.d
-include $(BUILD)/cudaprof.d

clean:
	rm -rf $(BUILD)

.PHONY: all clean format
