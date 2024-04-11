MAKEFLAGS+=-r -j

UNAME=$(shell uname)

NVCC?=nvcc

BUILD=build

SOURCES=$(wildcard src/*.c)

ifeq ($(UNAME),Darwin)
  SOURCES+=$(wildcard src/*.m)
  SOURCES+=$(wildcard src/*.metal)
endif

ifneq ($(UNAME),Darwin)
SOURCES+=$(wildcard src/*.cu)
endif

OBJECTS=$(SOURCES:%=$(BUILD)/%.o)
BINARY=$(BUILD)/run

CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math
LDFLAGS=-lm

ifeq ($(UNAME),Darwin)
  ifneq (,$(wildcard /opt/homebrew/opt/libomp))
    CFLAGS+=-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS+=-L/opt/homebrew/opt/libomp/lib -lomp
  endif
  LDFLAGS+=-framework Metal -framework Foundation
  METALFLAGS=-std=metal3.0 -O2
else
  CFLAGS+=-fopenmp -mf16c -mavx2 -mfma
  LDFLAGS+=-fopenmp
endif

ifneq ($(UNAME),Darwin)
  LDFLAGS+=-lcudart
endif

ifneq (,$(wildcard /usr/local/cuda))
  LDFLAGS+=-L/usr/local/cuda/lib64
endif

CUFLAGS+=-g -O2 -lineinfo
CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions

ifeq ($(CUARCH),)
  CUFLAGS+=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 2
else
  CUFLAGS+=-arch=$(CUARCH)
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

$(BUILD)/%.m.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.metal.o: %.metal
	@mkdir -p $(dir $@)
	xcrun metal $< $(METALFLAGS) -c -MMD -MP -o $@.ir
	xcrun metallib -o $@.metallib $@.ir
	xxd -i -n $(basename $(notdir $<))_metallib $@.metallib > $@.c
	$(CC) $@.c -c -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(BUILD)/cudabench.d
-include $(BUILD)/cudaprof.d

clean:
	rm -rf $(BUILD)

.PHONY: all clean format
