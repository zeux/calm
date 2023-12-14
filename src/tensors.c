#include "tensors.h"

#include "jsmn.h"

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>

static bool json_equal(const char* json, const jsmntok_t* tok, const char* str) {
	assert(tok->type == JSMN_STRING);
	size_t len = tok->end - tok->start;
	return len == strlen(str) && memcmp(json + tok->start, str, len) == 0;
}

static void json_copy(char* buf, size_t buf_size, const char* json, const jsmntok_t* tok) {
	assert(tok->type == JSMN_STRING || tok->type == JSMN_PRIMITIVE);
	size_t len = tok->end - tok->start;
	size_t res = len < buf_size ? len : buf_size - 1;
	memcpy(buf, json + tok->start, res);
	buf[res] = 0;
}

static long long json_llong(const char* json, const jsmntok_t* tok) {
	if (tok->type != JSMN_PRIMITIVE) {
		return -1;
	}

	char tmp[128];
	json_copy(tmp, sizeof(tmp), json, tok);
	return atoll(tmp);
}

static bool validate_shape(enum DType dtype, int shape[4], size_t length) {
	size_t expected_length = 1;
	int max_elements = INT_MAX;

	for (int i = 0; i < 4; ++i) {
		int dim = shape[i] == 0 ? 1 : shape[i];

		if (dim < 0 || dim > max_elements) {
			return false;
		}

		expected_length *= dim;
		max_elements /= dim;
	}

	size_t element_size = (dtype == dt_f32) ? sizeof(float) : (dtype == dt_f16) ? sizeof(uint16_t)
	                                                                            : sizeof(uint8_t);

	return expected_length * element_size == length;
}

static int parse_tensor(struct Tensor* tensor, void* bytes, size_t bytes_size, const char* json, const jsmntok_t* tokens, int toki) {
	assert(tokens[toki].type == JSMN_STRING);
	json_copy(tensor->name, sizeof(tensor->name), json, &tokens[toki]);
	toki++;

	if (tokens[toki].type != JSMN_OBJECT) {
		return -1;
	}

	int n_keys = tokens[toki].size;
	toki++;

	size_t length = 0;

	for (int i = 0; i < n_keys; ++i) {
		const jsmntok_t* key = &tokens[toki];

		if (json_equal(json, key, "dtype") && tokens[toki + 1].type == JSMN_STRING) {
			if (json_equal(json, &tokens[toki + 1], "F32")) {
				tensor->dtype = dt_f32;
			} else if (json_equal(json, &tokens[toki + 1], "F16")) {
				tensor->dtype = dt_f16;
			} else if (json_equal(json, &tokens[toki + 1], "U8")) {
				tensor->dtype = dt_u8;
			} else {
				return -1;
			}
			toki += 2;
		} else if (json_equal(json, key, "shape") && tokens[toki + 1].type == JSMN_ARRAY && tokens[toki + 1].size <= 4) {
			int shape_len = tokens[toki + 1].size;
			for (int j = 0; j < shape_len; ++j) {
				tensor->shape[j] = json_llong(json, &tokens[toki + 2 + j]);
				if (tensor->shape[j] < 0) {
					return -1;
				}
			}
			toki += 2 + shape_len;
		} else if (json_equal(json, key, "data_offsets") && tokens[toki + 1].type == JSMN_ARRAY && tokens[toki + 1].size == 2) {
			long long start = json_llong(json, &tokens[toki + 2]);
			long long end = json_llong(json, &tokens[toki + 3]);
			toki += 4;

			if (start < 0 || end <= start || end > bytes_size) {
				return -1;
			}

			tensor->data = (char*)bytes + start;
			length = end - start;
		} else {
			return -1;
		}
	}

	if (!validate_shape(tensor->dtype, tensor->shape, length)) {
		return -1;
	}

	return toki;
}

int tensors_parse(struct Tensors* tensors, void* data, size_t size) {
	if (size < sizeof(uint64_t)) {
		return -1;
	}

	uint64_t json_size = *(uint64_t*)data;
	if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
		return -1;
	}

	char* json = (char*)data + sizeof(uint64_t);
	void* bytes = (char*)data + sizeof(uint64_t) + json_size;
	size_t bytes_size = size - sizeof(uint64_t) - json_size;

	jsmn_parser parser;
	jsmntok_t tokens[16384];
	int tokres = jsmn_parse(&parser, json, json_size, tokens, sizeof(tokens) / sizeof(tokens[0]));

	if (tokres <= 0 || tokens[0].type != JSMN_OBJECT) {
		return -1;
	}

	int toki = 1;
	while (toki < tokres) {
		if (json_equal(json, &tokens[toki], "__metadata__") && tokens[toki + 1].type == JSMN_OBJECT) {
			int n_keys = tokens[toki + 1].size;
			toki += 2;

			for (int k = 0; k < n_keys; ++k) {
				assert(tokens[toki].type == JSMN_STRING);
				if (tokens[toki + 1].type != JSMN_STRING) {
					return -1;
				}

				if (tensors->n_metadata >= sizeof(tensors->metadata) / sizeof(tensors->metadata[0])) {
					return -1;
				}
				struct Metadata* metadata = &tensors->metadata[tensors->n_metadata++];

				json_copy(metadata->key, sizeof(metadata->key), json, &tokens[toki]);
				json_copy(metadata->value, sizeof(metadata->value), json, &tokens[toki + 1]);
				toki += 2;
			}
		} else {
			struct Tensor tensor = {};
			toki = parse_tensor(&tensor, bytes, bytes_size, json, tokens, toki);
			if (toki < 0) {
				return -1;
			}

			if (tensors->n_tensors >= sizeof(tensors->tensors) / sizeof(tensors->tensors[0])) {
				return -1;
			}
			tensors->tensors[tensors->n_tensors++] = tensor;
		}
	}

	return 0;
}

int tensors_open(struct Tensors* tensors, const char* filename) {
	int fd = open(filename, O_RDONLY);
	if (fd == -1) {
		return -1;
	}

	struct stat st;
	if (fstat(fd, &st) != 0) {
		close(fd);
		return -1;
	}

	size_t size = st.st_size;
	void* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (data == MAP_FAILED) {
		close(fd);
		return -1;
	}

	close(fd); // fd can be closed after mmap returns without invalidating the mapping

	if (tensors_parse(tensors, data, size) != 0) {
		munmap(data, size);
		return -2;
	}

	tensors->data = data;
	tensors->size = size;

	return 0;
}

void tensors_close(struct Tensors* tensors) {
	munmap(tensors->data, tensors->size);
}

struct Tensor* tensors_find(struct Tensors* tensors, const char* name, int layer) {
	char key[128];
	snprintf(key, sizeof(key), name, layer);

	for (int i = 0; i < tensors->n_tensors; ++i) {
		if (strcmp(tensors->tensors[i].name, key) == 0) {
			return &tensors->tensors[i];
		}
	}
	return NULL;
}

void* tensors_get(struct Tensors* tensors, const char* name, int layer, enum DType dtype, int shape[4]) {
	struct Tensor* tensor = tensors_find(tensors, name, layer);
	if (tensor == NULL) {
		fprintf(stderr, "FATAL: Tensor not found: %s\n", name);
		assert(false);
		return NULL;
	}

	if (tensor->dtype != dtype || memcmp(tensor->shape, shape, sizeof(tensor->shape)) != 0) {
		fprintf(stderr, "FATAL: Tensor mismatch: %s\n", name);
		fprintf(stderr, "  Expected: dtype=%d shape=[%d,%d,%d,%d]\n", dtype, shape[0], shape[1], shape[2], shape[3]);
		fprintf(stderr, "  Actual:   dtype=%d shape=[%d,%d,%d,%d]\n", tensor->dtype, tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]);
		assert(false);
		return NULL;
	}

	return tensor->data;
}

const char* tensors_metadata_find(struct Tensors* tensors, const char* name) {
	for (int i = 0; i < tensors->n_metadata; ++i) {
		if (strcmp(tensors->metadata[i].key, name) == 0) {
			return tensors->metadata[i].value;
		}
	}
	return NULL;
}

const char* tensors_metadata(struct Tensors* tensors, const char* name) {
	const char* res = tensors_metadata_find(tensors, name);
	if (res == NULL) {
		fprintf(stderr, "FATAL: Metadata not found: %s\n", name);
		assert(false);
	}
	return res;
}

#ifdef FUZZING
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
	struct Tensors tensors = {};
	tensors_parse(&tensors, (void*)data, size);
	return 0;
}
#endif
