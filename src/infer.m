#include "model.h"

#include <Metal/Metal.h>

extern unsigned char infer_metallib[];
extern unsigned int infer_metallib_len;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> kernels[256];

static void dispatch(id<MTLComputeCommandEncoder> encoder, const char* name, const char* variant, unsigned int thread_groups, unsigned int thread_group_size, void* params, size_t params_size, void** buffers, size_t buffer_count) {
	char expected[256];
	strcpy(expected, name);
	if (variant) {
		strcat(expected, "_");
		strcat(expected, variant);
	}

	id<MTLComputePipelineState> state = nil;
	for (size_t i = 0; kernels[i]; ++i) {
		if (strcmp(kernels[i].label.UTF8String, expected) == 0) {
			state = kernels[i];
			break;
		}
	}
	assert(state);

	[encoder setComputePipelineState:state];
	[encoder setBytes:params length:params_size atIndex:0];
	for (size_t i = 0; i < buffer_count; ++i) {
		[encoder setBuffer:buffers[i] offset:0 atIndex:i+1];
	}

	[encoder dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
}

void init_metal(void) {
	NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
	assert(devices.count > 0);

	device = devices[0];
	queue = [device newCommandQueue];

	dispatch_data_t lib_data = dispatch_data_create(infer_metallib, infer_metallib_len, dispatch_get_main_queue(), ^{});

	NSError *error = nil;
	id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
	assert(library);

	NSArray<NSString*>* functions = library.functionNames;
	for (size_t i = 0; i < functions.count; i++) {
		MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
		descriptor.computeFunction = [library newFunctionWithName:functions[i]];
		descriptor.label = functions[i];

		id<MTLComputePipelineState> computePipelineState = [device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
		assert(computePipelineState);
		kernels[i] = computePipelineState;
	}
}

void* upload_metal(void* host, size_t size) {
	assert(device);
	id<MTLBuffer> buffer = [device newBufferWithBytes:host length:size options:MTLResourceStorageModeShared];
	return buffer;
}

static void* newbuffer(size_t size) {
	return size == 0 ? nil : [device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

void prepare_metal(struct Transformer* transformer) {
	struct Config* config = &transformer->config;
	struct RunState* state = &transformer->state;

	assert(device);
	printf("# Metal: %s\n", device.name.UTF8String);

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int q_dim = config->head_dim * config->n_heads;
	int kv_dim = config->head_dim * config->n_kv_heads;

	state->x = (float*)newbuffer(dim * sizeof(float));
	state->hb = (float*)newbuffer(hidden_dim * sizeof(float));
	state->he = (float*)newbuffer(config->n_experts_ac * hidden_dim * sizeof(float));
	state->q = (float*)newbuffer(q_dim * sizeof(float));
	state->att = (float*)newbuffer(config->n_heads * config->seq_len * 2 * sizeof(float));

	assert(state->kvbits == 8 || state->kvbits == 16);
	state->key_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));
	state->value_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)newbuffer(config->vocab_size * sizeof(float));
}

float* forward_metal(struct Transformer* transformer, int token, int pos, unsigned flags) {
	struct Config* p = &transformer->config;
	struct Weights* w = &transformer->weights;
	struct RunState* s = &transformer->state;

	// a few convenience variables
	float* x = s->x;
	int dim = p->dim;
	int hidden_dim = p->hidden_dim;
	int kv_dim = p->head_dim * p->n_kv_heads;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);

	// begin command recording
	id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
	id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

	// copy the token embedding into x
	assert(token < p->vocab_size);
	dispatch(encoder, "embed", "half", dim / 32, 32, (int[]){ token * dim }, sizeof(int), (void*[]){ x, w->token_embedding_table }, 2);

	// classifier into logits
	if ((flags & FF_UPDATE_KV_ONLY) == 0) {
		dispatch(encoder, "output", "half", p->vocab_size, 32, (int[]) { dim }, sizeof(int), (void*[]) { s->logits, x, w->wcls }, 3);
	}

	// submit commands and wait
	[encoder endEncoding];
	[commands commit];
	[commands waitUntilCompleted];

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	return [(id<MTLBuffer>)s->logits contents];
}