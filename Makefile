MAKEFLAGS+=-r -j

NVCC?=nvcc

BUILD=build

SOURCES=$(wildcard src/*.c) $(wildcard src/*.cu)
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

BINARY=$(BUILD)/run

CFLAGS=-g -Wall -Werror -O3 -ffast-math -Iextern -fopenmp -mf16c -mavx2
CUFLAGS=-g -O2
LDFLAGS=-lm -fopenmp -lcudart

all: $(BINARY)

format:
	clang-format -i src/*.c src/*.h src/*.cu

$(BUILD)/fuzz-tensors: src/tensors.c
	clang $(CFLAGS) -DFUZZING -O1 -fsanitize=address,fuzzer -o $@ $^

$(BUILD)/perftest-cuda: src/infer-cuda.cu
	$(NVCC) $(CUFLAGS) -DCUDA_PERFTEST -o $@ $^

$(BINARY): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format
