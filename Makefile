MAKEFLAGS+=-r -j

BUILD=build

SOURCES=$(wildcard src/*.c)
OBJECTS=$(SOURCES:%=$(BUILD)/%.o)

BINARY=$(BUILD)/run

CFLAGS=-g -Wall -Werror -O3 -ffast-math -Iextern -fopenmp -mf16c -mavx2
LDFLAGS=-lm -fopenmp

all: $(BINARY)

format:
	clang-format -i src/*.c src/*.h

$(BUILD)/fuzz-tensors: src/tensors.c
	clang $(CFLAGS) -DFUZZING -O1 -fsanitize=address,fuzzer -o $@ $^

$(BINARY): $(OBJECTS)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $< $(CFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format
