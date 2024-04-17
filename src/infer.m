#include "model.h"

#include <Metal/Metal.h>

extern unsigned char infer_metallib[];
extern unsigned int infer_metallib_len;

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> kernels[256];
static const char* kernel_names[256];

static MTLCaptureManager* capture;

static void dispatch(id<MTLComputeCommandEncoder> encoder, const char* name, const char* variant, unsigned int threadgroups, unsigned int threadgroup_size, unsigned int threadgroup_smem, void* params, size_t params_size, void** buffers, size_t buffer_count) {
	char expected[256];
	strcpy(expected, name);
	if (variant) {
		strcat(expected, "_");
		strcat(expected, variant);
	}

	id<MTLComputePipelineState> state = nil;
	for (size_t i = 0; kernels[i]; ++i) {
		if (strcmp(kernel_names[i], expected) == 0) {
			state = kernels[i];
			break;
		}
	}
	assert(state);
	assert(state.maxTotalThreadsPerThreadgroup >= threadgroup_size);

	static const NSUInteger offsets[16] = {};

	[encoder setComputePipelineState:state];
	[encoder setBytes:params length:params_size atIndex:0];
	[encoder setBuffers:(const id<MTLBuffer>*)buffers offsets:offsets withRange:NSMakeRange(1, buffer_count)];
	[encoder setThreadgroupMemoryLength:threadgroup_smem atIndex:0];
	[encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
}

void init_metal(void) {
	NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
	assert(devices.count > 0);

	device = devices[0];
	queue = [device newCommandQueue];

	dispatch_data_t lib_data = dispatch_data_create(infer_metallib, infer_metallib_len, dispatch_get_main_queue(), ^{
	                                                                                    });

	NSError* error = nil;
	id<MTLLibrary> library = [device newLibraryWithData:lib_data error:&error];
	assert(library);

	NSArray<NSString*>* functions = library.functionNames;
	for (size_t i = 0; i < functions.count; i++) {
		id<MTLFunction> function = [library newFunctionWithName:functions[i]];
		id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
		assert(state);
		kernels[i] = state;
		kernel_names[i] = [functions[i] UTF8String];
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
	struct Weights* weights = &transformer->weights;
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
	state->exp = (float*)newbuffer(((config->n_experts_ac ? config->n_experts_ac : 1) * 2) * sizeof(float));

	assert(state->kvbits == 8 || state->kvbits == 16);
	state->key_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));
	state->value_cache = newbuffer((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)newbuffer(config->vocab_size * sizeof(float));

	float* bqkv = (float*)newbuffer((q_dim + kv_dim * 2) * sizeof(float));

	for (int l = 0; l < config->n_layers; ++l) {
		if (weights->bqkv[l] == NULL) {
			weights->bqkv[l] = bqkv;
		}
	}

	if (config->n_experts == 0) {
		// setup expert buffer to always point to the first (and only) expert
		float* moe = [(id<MTLBuffer>)state->exp contents];
		moe[0] = 1.0f;
		moe[1] = 0.0f;
	}

	if (weights->dbits == 4) {
		id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
		id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

		dispatch(encoder, "prepare_gf4", NULL, dim * config->vocab_size / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->token_embedding_table}, 1);

		for (int l = 0; l < config->n_layers; ++l) {
			dispatch(encoder, "prepare_gf4", NULL, q_dim * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->wq[l]}, 1);
			dispatch(encoder, "prepare_gf4", NULL, kv_dim * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->wk[l]}, 1);
			dispatch(encoder, "prepare_gf4", NULL, kv_dim * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->wv[l]}, 1);
			dispatch(encoder, "prepare_gf4", NULL, dim * q_dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->wo[l]}, 1);

			int n_experts = config->n_experts ? config->n_experts : 1;

			dispatch(encoder, "prepare_gf4", NULL, n_experts * hidden_dim * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->w1[l]}, 1);
			dispatch(encoder, "prepare_gf4", NULL, n_experts * dim * hidden_dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->w2[l]}, 1);
			dispatch(encoder, "prepare_gf4", NULL, n_experts * hidden_dim * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->w3[l]}, 1);

			if (weights->moegate[l]) {
				dispatch(encoder, "prepare_gf4", NULL, config->n_experts * dim / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->moegate[l]}, 1);
			}
		}

		if (weights->wcls != weights->token_embedding_table) {
			dispatch(encoder, "prepare_gf4", NULL, dim * config->vocab_size / 256, 32, 0, (int[]){0}, sizeof(int), (void*[]){weights->wcls}, 1);
		}

		[encoder endEncoding];
		[commands commit];
		[commands waitUntilCompleted];
	}

	const char* capenv = getenv("MTL_CAPTURE_ENABLED");
	if (capenv && atoi(capenv)) {
		capture = [MTLCaptureManager sharedCaptureManager];
		assert(capture);

		NSString* path = @"calm.gputrace";

		MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
		desc.captureObject = queue;
		desc.destination = MTLCaptureDestinationGPUTraceDocument;
		desc.outputURL = [NSURL fileURLWithPath:path];

		NSError* error = nil;
		[[NSFileManager defaultManager] removeItemAtPath:path error:&error];

		BOOL started = [capture startCaptureWithDescriptor:desc error:&error];
		assert(started);

		NSLog(@"Capturing first token to %@", desc.outputURL);
	}
}

struct SinkArgs {
	int kv_dim;
	int head_dim;
	int rotary_dim;

	int kv_sink;
	int seq_len;

	float theta_log2;
};

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
	assert(s->kvbits == 16); // TODO

	const char* dvar = w->dbits == 16 ? "half" : (w->dbits == 8 ? "fp8" : (w->dbits == 4 ? "gf4" : "?"));
	const char* kvar = "half";
	const char* nvar = w->dbits == 4 ? "half" : "float";

	char dkvar[32];
	snprintf(dkvar, sizeof(dkvar), "%s_%s", dvar, kvar);

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);

	// begin command recording
	id<MTLCommandBuffer> commands = [queue commandBufferWithUnretainedReferences];
	id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];

	// copy the token embedding into x
	assert(token < p->vocab_size);
	dispatch(encoder, "embed", dvar, dim / 32, 32, 0, (int[]){token * dim}, sizeof(int), (void*[]){x, w->token_embedding_table}, 2);

	// rotate sink tokens forward to keep pace with non-sink tokens
	if (kv_sink > 0) {
		dispatch(encoder, "rotate_sink", kvar, (kv_sink * kv_dim / 64) * p->n_layers, 32, 0, &(struct SinkArgs){kv_dim, p->head_dim, p->rotary_dim, kv_sink, p->seq_len, log2(p->rope_theta)}, sizeof(struct SinkArgs), (void*[]){s->key_cache}, 1);
	}

	// forward all the layers
	for (int l = 0; l < p->n_layers; ++l) {
		size_t loff = (size_t)l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		// pre-attention rmsnorm
		dispatch(encoder, "rmsnorm", nvar, 1, 1024, 0, &(struct NormArgs){dim, p->norm_eps, p->norm_ln}, sizeof(struct NormArgs), (void*[]){s->xb, x, w->rms_att_weight[l]}, 3);

		// qkv
		dispatch(encoder, "qkv", dkvar, (q_dim + kv_dim * 2) / 2, 32, 0, &(struct QkvArgs){dim, q_dim, kv_dim, p->head_dim, p->rotary_dim, pos, kv_pos, p->seq_len, loff, p->qkv_clip, log2(p->rope_theta)}, sizeof(struct QkvArgs), (void*[]){s->xb, s->q, s->key_cache, s->value_cache, w->wq[l], w->wk[l], w->wv[l], w->bqkv[l]}, 8);

		// attn score
		int kv_lent = (kv_len + 7) / 8;

		dispatch(encoder, "attn_score", kvar, kv_lent * p->n_heads, 32, 0, &(struct AttnArgs){p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff}, sizeof(struct AttnArgs), (void*[]){s->att, s->q, s->key_cache}, 3);

		// attn softmax
		dispatch(encoder, "attn_softmax", NULL, p->n_heads, 1024, 0, &(struct AttnArgs){p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff}, sizeof(struct AttnArgs), (void*[]){s->att}, 1);

		// attn mix
		dispatch(encoder, "attn_mix", kvar, q_dim, 32, 0, &(struct AttnArgs){p->seq_len, kv_len, p->head_dim, kv_mul, p->n_heads, loff}, sizeof(struct AttnArgs), (void*[]){s->q, s->att, s->value_cache}, 3);

		// attn out
		dispatch(encoder, "attn_out", dvar, dim, 32, 0, (int[]){q_dim}, sizeof(int), (void*[]){x, s->q, w->wo[l]}, 3);

		if (!p->norm_par) {
			// post-attention rmsnorm
			dispatch(encoder, "rmsnorm", nvar, 1, 1024, 0, &(struct NormArgs){dim, p->norm_eps, p->norm_ln}, sizeof(struct NormArgs), (void*[]){s->xb, x, w->rms_ffn_weight[l]}, 3);
		}

		// moe gate
		if (p->n_experts) {
			dispatch(encoder, "moe_gate", dvar, 1, p->n_experts * 32, 0, (int[]){dim, p->n_experts, p->n_experts_ac}, sizeof(int) * 3, (void*[]){s->exp, s->xb, w->moegate[l]}, 3);
		}

		// ffn
		float* hb = p->n_experts ? s->he : s->hb;
		int n_experts_ac = p->n_experts_ac ? p->n_experts_ac : 1;

		dispatch(encoder, p->act_gelu ? "ffn13_gelu" : "ffn13_silu", dvar, n_experts_ac * hidden_dim, 32, 0, (int[]){dim, hidden_dim}, sizeof(int) * 2, (void*[]){hb, s->xb, s->exp, w->w1[l], w->w3[l]}, 5);
		dispatch(encoder, "ffn2", dvar, n_experts_ac * dim, 32, 0, (int[]){hidden_dim, dim}, sizeof(int) * 2, (void*[]){x, hb, s->exp, w->w2[l]}, 4);
	}

	// classifier into logits
	if ((flags & FF_UPDATE_KV_ONLY) == 0) {
		dispatch(encoder, "rmsnorm", nvar, 1, 1024, 0, &(struct NormArgs){dim, p->norm_eps, p->norm_ln}, sizeof(struct NormArgs), (void*[]){s->xb, x, w->rms_final_weight}, 3);
		dispatch(encoder, "output", dvar, p->vocab_size, 32, 0, (int[]){dim}, sizeof(int), (void*[]){s->logits, s->xb, w->wcls}, 3);
	}

	// submit commands and wait
	[encoder endEncoding];
	[commands commit];
	[commands waitUntilCompleted];

	if (capture) {
		[capture stopCapture];
		capture = nil;
	}

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	return [(id<MTLBuffer>)s->logits contents];
}