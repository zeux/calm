MAKEFLAGS+=-r -j

NVCC?=nvcc

BUILD=build

SOURCES=$(wildcard src/*.c) $(wildcard src/*.cu)
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

BINARY=$(BUILD)/run

CFLAGS=-g -Wall -Werror -O3 -ffast-math -Iextern -fopenmp -mf16c -mavx2
CUFLAGS=-g -O2 -arch compute_80
LDFLAGS=-lm -fopenmp -lcudart

ifneq (,$(wildcard /usr/local/cuda))
    CFLAGS+=-I/usr/local/cuda/include
    LDFLAGS+=-L/usr/local/cuda/lib64
endif

all: $(BINARY)

format:
	clang-format -i src/* tools/*.cu

$(BUILD)/fuzz-tensors: src/tensors.c
	clang $(CFLAGS) -DFUZZING -O1 -fsanitize=address,fuzzer -o $@ $^

$(BUILD)/perftest-cuda: tools/perftest.cu
	$(NVCC) $< $(CUFLAGS) -MMD -MP -o $@

$(BINARY): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(BUILD)/perftest-cuda.d

clean:
	rm -rf $(BUILD)

.PHONY: all clean format
