#pragma once

#include <stddef.h>

enum DType {
	dt_f32,
	dt_f16,
	dt_bf16,
	dt_f8e5m2,
	dt_f8e4m3,
	dt_i32,
	dt_i16,
	dt_i8,
	dt_u8,
};

struct Tensor {
	char* name;
	enum DType dtype;
	int shape[4];
	void* data;
};

struct Metadata {
	char* key;
	char* value;
};

struct Tensors {
	void* data;
	size_t size;

	char* json;

	struct Metadata metadata[128];
	int n_metadata;

	struct Tensor tensors[1024];
	int n_tensors;
};

int tensors_parse(struct Tensors* tensors, void* data, size_t size);

int tensors_open(struct Tensors* tensors, const char* filename);
void tensors_close(struct Tensors* tensors);

struct Tensor* tensors_find(struct Tensors* tensors, const char* name, int layer);
void* tensors_get(struct Tensors* tensors, const char* name, int layer, enum DType dtype, int shape[4]);

const char* tensors_metadata_find(struct Tensors* tensors, const char* name);
const char* tensors_metadata(struct Tensors* tensors, const char* name);
