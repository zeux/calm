#include "tensors.h"

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

static char* json_skipws(char* json) {
	while (*json == ' ' || *json == '\t' || *json == '\n' || *json == '\r') {
		json++;
	}
	return json;
}

static char* json_string(char* json, char** res) {
	if (*json != '"') {
		return NULL;
	}
	json++;

	*res = json;
	while (*json != '"') {
		if (*json == 0 || *json == '\\') {
			return NULL;
		}
		json++;
	}

	*json = 0;
	json++;

	return json_skipws(json);
}

static char* json_key(char* json, char** key) {
	json = json_string(json, key);
	if (json == NULL) {
		return NULL;
	}
	if (*json != ':') {
		return NULL;
	}
	return json_skipws(json + 1);
}

static char* json_llong(char* json, long long* res) {
	char* end;
	*res = strtoll(json, &end, 10);
	if (end == json) {
		return NULL;
	}
	end = json_skipws(end);
	if (*end != ',' && *end != ']') {
		return NULL;
	}
	return end;
}

static int json_dtype(const char* str, enum DType* dtype, int* dsize) {
	if (strcmp(str, "F32") == 0) {
		*dtype = dt_f32;
		*dsize = 4;
	} else if (strcmp(str, "F16") == 0) {
		*dtype = dt_f16;
		*dsize = 2;
	} else if (strcmp(str, "BF16") == 0) {
		*dtype = dt_bf16;
		*dsize = 2;
	} else if (strcmp(str, "F8_E5M2") == 0) {
		*dtype = dt_f8e5m2;
		*dsize = 1;
	} else if (strcmp(str, "F8_E4M3") == 0) {
		*dtype = dt_f8e4m3;
		*dsize = 1;
	} else if (strcmp(str, "I32") == 0) {
		*dtype = dt_i32;
		*dsize = 4;
	} else if (strcmp(str, "I16") == 0) {
		*dtype = dt_i16;
		*dsize = 2;
	} else if (strcmp(str, "I8") == 0) {
		*dtype = dt_i8;
		*dsize = 1;
	} else if (strcmp(str, "U8") == 0) {
		*dtype = dt_u8;
		*dsize = 1;
	} else {
		return -1;
	}

	return 0;
}

static bool validate_shape(int dsize, int shape[4], size_t length) {
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

	return expected_length * dsize == length;
}

static char* parse_tensor(struct Tensor* tensor, void* bytes, size_t bytes_size, char* name, char* s) {
	tensor->name = name;

	if (*s != '{') {
		return NULL;
	}
	s = json_skipws(s + 1);

	size_t length = 0;
	int dsize = 0;

	while (*s && *s != '{') {
		char* key;
		s = json_key(s, &key);
		if (!s) {
			return NULL;
		}

		if (strcmp(key, "dtype") == 0) {
			char* val;
			s = json_string(s, &val);
			if (!s) {
				return NULL;
			}
			if (json_dtype(val, &tensor->dtype, &dsize) != 0) {
				return NULL;
			}
		} else if (strcmp(key, "shape") == 0) {
			if (*s != '[') {
				return NULL;
			}
			s = json_skipws(s + 1);

			for (int j = 0; j < 4; ++j) {
				long long val;
				s = json_llong(s, &val);
				if (!s) {
					return NULL;
				}
				if (val < 0 || val > INT_MAX) {
					return NULL;
				}
				tensor->shape[j] = (int)val;

				if (*s == ']') {
					break;
				}
				if (*s != ',') {
					return NULL;
				}
				s = json_skipws(s + 1);
			}

			if (*s != ']') {
				return NULL;
			}
			s = json_skipws(s + 1);
		} else if (strcmp(key, "data_offsets") == 0) {
			if (*s != '[') {
				return NULL;
			}
			s = json_skipws(s + 1);

			long long start;
			s = json_llong(s, &start);

			if (*s != ',') {
				return NULL;
			}
			s = json_skipws(s + 1);

			long long end;
			s = json_llong(s, &end);

			if (*s != ']') {
				return NULL;
			}
			s = json_skipws(s + 1);

			if (start < 0 || end <= start || end > bytes_size) {
				return NULL;
			}

			tensor->data = (char*)bytes + start;
			length = end - start;
		} else {
			return NULL;
		}

		if (*s == '}') {
			break;
		}
		if (*s != ',') {
			return NULL;
		}
		s = json_skipws(s + 1);
	}

	if (!validate_shape(dsize, tensor->shape, length)) {
		return NULL;
	}

	if (*s != '}') {
		return NULL;
	}
	s = json_skipws(s + 1);

	return s;
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

	char* jsoncopy = malloc(json_size + 1);
	if (!jsoncopy) {
		return -1;
	}
	memcpy(jsoncopy, json, json_size);
	jsoncopy[json_size] = 0;

	tensors->json = jsoncopy;

	char* s = jsoncopy;

	if (*s != '{') {
		return -1;
	}
	s = json_skipws(s + 1);

	while (*s && *s != '}') {
		char* key;
		s = json_key(s, &key);
		if (!s) {
			return -1;
		}

		if (strcmp(key, "__metadata__") == 0) {
			if (*s != '{') {
				return -1;
			}
			s = json_skipws(s + 1);

			while (*s && *s != '}') {
				char* metakey;
				s = json_key(s, &metakey);
				if (!s) {
					return -1;
				}
				char* metaval;
				s = json_string(s, &metaval);
				if (!s) {
					return -1;
				}

				if (tensors->n_metadata >= sizeof(tensors->metadata) / sizeof(tensors->metadata[0])) {
					return -1;
				}
				struct Metadata* metadata = &tensors->metadata[tensors->n_metadata++];

				metadata->key = metakey;
				metadata->value = metaval;

				if (*s == '}') {
					break;
				}
				if (*s != ',') {
					return -1;
				}
				s = json_skipws(s + 1);
			}
			if (*s != '}') {
				return -1;
			}
			s = json_skipws(s + 1);
		} else {
			struct Tensor tensor = {};
			s = parse_tensor(&tensor, bytes, bytes_size, key, s);
			if (s == NULL) {
				return -1;
			}

			if (tensors->n_tensors >= sizeof(tensors->tensors) / sizeof(tensors->tensors[0])) {
				return -1;
			}
			tensors->tensors[tensors->n_tensors++] = tensor;
		}

		if (*s == '}') {
			break;
		}
		if (*s != ',') {
			return -1;
		}
		s = json_skipws(s + 1);
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

	posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
	close(fd); // fd can be closed after mmap returns without invalidating the mapping

	if (tensors_parse(tensors, data, size) != 0) {
		free(tensors->json);
		munmap(data, size);
		return -2;
	}

	tensors->data = data;
	tensors->size = size;

	return 0;
}

void tensors_close(struct Tensors* tensors) {
	free(tensors->json);
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
	tensors_close(&tensors);
	return 0;
}
#endif
