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
	state->xb = (float*)newbuffer(dim * sizeof(float));
	state->hb = (float*)newbuffer(hidden_dim * sizeof(float));
	state->he = (float*)newbuffer(config->n_experts_ac * hidden_dim * sizeof(float));
	state->q = (float*)newbuffer(q_dim * sizeof(float));
	state->att = (float*)newbuffer(config->n_heads * config->seq_len * sizeof(float));

	assert(state->kvbits == 8 || state->kvbits == 16);
	state->key_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));
	state->value_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)newbuffer(config->vocab_size * sizeof(float));
}

struct NormArgs {
	int size;
	float eps;
	bool ln;
};

struct QkvArgs {
	int dim;
	int q_dim;
	int kv_dim;
	int head_dim;
	int rotary_dim;

	int pos;
	int kv_pos;
	int seq_len;

	size_t loff;

	float qkv_clip;
	float theta_log2;
};

struct AttnArgs {
	int seq_len;
	int kv_len;
	int head_dim;
	int kv_mul;
	int n_heads;

	size_t loff;
};

float* forward_metal(struct Transformer* transformer, int token, int pos, unsigned flags) {
	struct Config* p = &transformer->config;
	struct Weights* w = &transformer->weights;
	struct RunState* s = &transformer->state;

	// a few convenience variables
	float* x = s->x;
	int dim = p->dim;
	int hidden_dim = p->hidden_dim;
	int kv_dim = p->head_dim * p->n_kv_heads;
	int q_dim = p->head_dim * p->n_heads;
	int kv_mul = p->n_heads / p->n_kv_heads;
	assert(w->dbits == 16); // TODO
	assert(s->kvbits == 16); // TODO

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;
	(void)kv_len; // TODO

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);

	// begin command recording
	id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
	id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

	// copy the token embedding into x
	assert(token < p->vocab_size);
	dispatch(encoder, "embed", "half", dim / 32, 32, (int[]){ token * dim }, sizeof(int), (void*[]){ x, w->token_embedding_table }, 2);

	for (int l = 0; l < p->n_layers; ++l) {
		size_t loff = (size_t)l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		// pre-attention rmsnorm
		dispatch(encoder, "rmsnorm", NULL, 1, 1024, &(struct NormArgs) { dim, p->norm_eps, p->norm_ln }, sizeof(struct NormArgs), (void*[]) { s->xb, x, w->rms_att_weight[l] }, 3);

		// qkv
		assert(w->bqkv[l] == NULL); // TODO

		dispatch(encoder, "qkv", "half_half", (q_dim + kv_dim * 2) / 2, 32, &(struct QkvArgs) { dim, q_dim, kv_dim, p->head_dim, p->rotary_dim, pos, kv_pos, p->seq_len, loff, p->qkv_clip, log2(p->rope_theta) }, sizeof(struct QkvArgs), (void*[]) { s->xb, s->q, s->key_cache, s->value_cache, w->wq[l], w->wk[l], w->wv[l] }, 7);

		// attn score
		int kv_lent = (kv_len + 7) / 8;

		dispatch(encoder, "attn_score", "half", kv_lent * p->n_heads, 32, &(struct AttnArgs) { p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff }, sizeof(struct AttnArgs), (void*[]) { s->att, s->q, s->key_cache }, 3);

		// attn softmax
		dispatch(encoder, "attn_softmax", NULL, p->n_heads, 1024, &(struct AttnArgs) { p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff }, sizeof(struct AttnArgs), (void*[]) { s->att }, 1);

		// attn mix
		dispatch(encoder, "attn_mix", "half", q_dim, 32, &(struct AttnArgs) { p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff }, sizeof(struct AttnArgs), (void*[]) { s->q, s->att, s->value_cache }, 3);

		// attn out
		dispatch(encoder, "attn_out", "half", dim, 32, (int[]) { q_dim }, sizeof(int), (void*[]) { x, s->q, w->wo[l] }, 3);

		if (!p->norm_par) {
			// post-attention rmsnorm
			dispatch(encoder, "rmsnorm", NULL, 1, 1024, &(struct NormArgs) { dim, p->norm_eps, p->norm_ln }, sizeof(struct NormArgs), (void*[]) { s->xb, x, w->rms_ffn_weight[l] }, 3);
		}

		assert(p->n_experts == 0); // TODO

		// ffn
		dispatch(encoder, p->act_gelu ? "ffn13_gelu" : "ffn13_silu", "half", hidden_dim, 32, (int[]) { dim }, sizeof(int), (void*[]) { s->hb, s->xb, w->w1[l], w->w3[l] }, 4);
		dispatch(encoder, "ffn2", "half", dim, 32, (int[]) { hidden_dim }, sizeof(int), (void*[]) { x, s->hb, w->w2[l] }, 3);
	}

	// classifier into logits
	if ((flags & FF_UPDATE_KV_ONLY) == 0) {
		dispatch(encoder, "rmsnorm", NULL, 1, 1024, &(struct NormArgs) { dim, p->norm_eps, p->norm_ln }, sizeof(struct NormArgs), (void*[]) { s->xb, x, w->rms_final_weight }, 3);
		dispatch(encoder, "output", "half", p->vocab_size, 32, (int[]) { dim }, sizeof(int), (void*[]) { s->logits, s->xb, w->wcls }, 3);
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