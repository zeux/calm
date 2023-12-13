#pragma once

#include <stddef.h>

enum DType {
	dt_f32,
	dt_f16,
};

struct Tensor {
	char name[64];
	enum DType dtype;
	int shape[4];
	void* data;
};

struct Metadata {
    char key[64];
    char value[64];
};

struct Tensors {
	void* data;
	size_t size;

    struct Metadata metadata[128];
    int n_metadata;

	struct Tensor tensors[1024];
	int n_tensors;
};

int tensors_open(struct Tensors* tensors, const char* filename);
void tensors_close(struct Tensors* tensors);

struct Tensor* tensors_find(struct Tensors* tensors, const char* name, int layer);
void* tensors_get(struct Tensors* tensors, const char* name, int layer, enum DType dtype, int shape[4]);

const char* tensors_metadata_find(struct Tensors* tensors, const char* name);
const char* tensors_metadata(struct Tensors* tensors, const char* name);
