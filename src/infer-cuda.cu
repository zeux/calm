#include "model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <cuda_fp16.h>

typedef __half cudtype_t;

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
			        cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUDA_SYNC() CUDA_CHECK(cudaDeviceSynchronize())

static void* cuda_devicecopy(void* host, size_t size) {
	void* device = NULL;
	CUDA_CHECK(cudaMalloc(&device, size));
	CUDA_CHECK(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
	return device;
}

static void* cuda_devicealloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, size));
	return ptr;
}

static void* cuda_hostalloc(size_t size) {
	void* ptr = NULL;
	CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
	return ptr;
}

extern "C" void prepare_cuda(struct Transformer* transformer) {
	struct Config* config = &transformer->config;
	struct Weights* weights = &transformer->weights;
	struct RunState* state = &transformer->state;

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (dtype_t*)cuda_devicecopy(weights->rms_att_weight[l], dim * sizeof(dtype_t));
		weights->rms_ffn_weight[l] = (dtype_t*)cuda_devicecopy(weights->rms_ffn_weight[l], dim * sizeof(dtype_t));

		weights->wq[l] = (dtype_t*)cuda_devicecopy(weights->wq[l], dim * dim * sizeof(dtype_t));
		weights->wk[l] = (dtype_t*)cuda_devicecopy(weights->wk[l], dim * kv_dim * sizeof(dtype_t));
		weights->wv[l] = (dtype_t*)cuda_devicecopy(weights->wv[l], dim * kv_dim * sizeof(dtype_t));
		weights->wo[l] = (dtype_t*)cuda_devicecopy(weights->wo[l], dim * dim * sizeof(dtype_t));

		weights->w1[l] = (dtype_t*)cuda_devicecopy(weights->w1[l], dim * hidden_dim * sizeof(dtype_t));
		weights->w2[l] = (dtype_t*)cuda_devicecopy(weights->w2[l], dim * hidden_dim * sizeof(dtype_t));
		weights->w3[l] = (dtype_t*)cuda_devicecopy(weights->w3[l], dim * hidden_dim * sizeof(dtype_t));
	}

	weights->rms_final_weight = (dtype_t*)cuda_devicecopy(weights->rms_final_weight, dim * sizeof(dtype_t));
	weights->token_embedding_table = (dtype_t*)cuda_devicecopy(weights->token_embedding_table, config->vocab_size * dim * sizeof(dtype_t));
	weights->wcls = (dtype_t*)cuda_devicecopy(weights->wcls, dim * config->vocab_size * sizeof(dtype_t));

	state->x = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb = (float*)cuda_devicealloc(dim * sizeof(float));
	state->hb = (float*)cuda_devicealloc(hidden_dim * sizeof(float));
	state->q = (float*)cuda_devicealloc(dim * sizeof(float));
	state->key_cache = (float*)cuda_devicealloc(config->n_layers * config->seq_len * kv_dim * sizeof(float));
	state->value_cache = (float*)cuda_devicealloc(config->n_layers * config->seq_len * kv_dim * sizeof(float));
	state->att = (float*)cuda_devicealloc(config->n_heads * config->seq_len * sizeof(float));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)cuda_hostalloc(config->vocab_size * sizeof(float));
}

__device__ static float warpreduce_sum(float v) {
	#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, mask);
	}
	return v;
}

__device__ static float warpreduce_max(float v) {
	#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
	}
	return v;
}

__device__ static float matmul(float* x, cudtype_t* w, int i, int n) {
	float val = 0.0f;
	for (int j = 0; j < n; j++) {
		val += float(w[i * n + j]) * x[j];
	}
	return val;
}

__global__ static void kernel_embed(float* o, cudtype_t* weight, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < size);

	o[i] = float(weight[i]);
}

__global__ static void kernel_rmsnorm(float* o, float* x, cudtype_t* weight, int size) {
	int i = threadIdx.x;

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += warpSize) {
		ss += x[j] * x[j];
	}

	// sum across threads
	ss = warpreduce_sum(ss);

	// compute scale
	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrtf(ss);

	// normalize and scale
	for (int j = i; j < size; j += warpSize) {
		o[j] = float(weight[j]) * (ss * x[j]);
	}
}

__global__ static void kernel_matmul_cls(float* xout, float* x, cudtype_t* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float val = matmul(x, w, i, n);

	xout[i] = val;
}

__global__ static void kernel_matmul_q(float* xout, float* x, cudtype_t* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float val = matmul(x, w, i, n);

	xout[i] = val;
}

__global__ static void kernel_matmul_kv(float* xout, float* x, cudtype_t* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float val = matmul(x, w, i, n);

	xout[i] = val;
}

__global__ static void kernel_matmul_attn(float* xout, float* x, cudtype_t* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float val = matmul(x, w, i, n);
	val += xout[i];

	xout[i] = val;
}

__global__ static void kernel_matmul_ffn13(float* xout, float* x, cudtype_t* w1, cudtype_t* w3, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float v1 = matmul(x, w1, i, n);
	float v3 = matmul(x, w3, i, n);

	// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	float val = v1;
	val *= 1.0f / (1.0f + expf(-v1));
	val *= v3;

	xout[i] = val;
}

__global__ static void kernel_matmul_ffn2(float* xout, float* x, cudtype_t* w, int n, int d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < d);

	float val = matmul(x, w, i, n);
	val += xout[i];

	xout[i] = val;
}

__global__ static void kernel_rope(float* vec, int head_size, int pos, float theta, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	assert(unsigned(i) < d);

	int head_dim = i % head_size;
	float freq = 1.0f / powf(theta, head_dim / (float)head_size);
	float val = pos * freq;
	float fcr = cosf(val);
	float fci = sinf(val);

	float v0 = vec[i];
	float v1 = vec[i + 1];
	vec[i] = v0 * fcr - v1 * fci;
	vec[i + 1] = v0 * fci + v1 * fcr;
}

__global__ static void kernel_attn_score(float* attb, float* qb, float* kb, int n_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int pos) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t > pos) {
		return;
	}

	int h = blockIdx.y;
	assert(unsigned(h) < n_heads);

	float* q = qb + h * head_size;
	float* k = kb + t * kv_dim + (h / kv_mul) * head_size;
	float* att = attb + h * seq_len;

	float score = 0.0f;
	for (int j = 0; j < head_size; j++) {
		score += q[j] * k[j];
	}
	score /= sqrtf(head_size);

	att[t] = score;
}

__global__ static void kernel_attn_softmax(float* attb, int n_heads, int seq_len, int pos) {
	int i = threadIdx.x;

	int h = blockIdx.y;
	assert(unsigned(h) < n_heads);

	float* att = attb + h * seq_len;

	// find max value per thread (for numerical stability)
	float max_val = 0.f;
	for (int j = i; j <= pos; j += warpSize) {
		max_val = max(max_val, att[j]);
	}

	// max across threads
	max_val = warpreduce_max(max_val);

	// exp and sum per thread
	float sum = 0.0f;
	for (int j = i; j <= pos; j += warpSize) {
		sum += expf(att[j] - max_val);
	}

	// sum across threads
	sum = warpreduce_sum(sum);

	// output normalized values
	for (int j = i; j <= pos; j += warpSize) {
		att[j] = expf(att[j] - max_val) / sum;
	}
}

__global__ static void kernel_attn_mix(float* xout, float* attb, float* valb, int n_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int pos) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(unsigned(i) < head_size);

	int h = blockIdx.y;
	assert(unsigned(h) < n_heads);

	float* att = attb + h * seq_len;
	float* val = valb + (h / kv_mul) * head_size;

	float res = 0.0f;
	for (int t = 0; t <= pos; t++) {
		res += att[t] * val[t * kv_dim + i];
	}

	xout[h * head_size + i] = res;
}

extern "C" float* forward_cuda(struct Transformer* transformer, int token, int pos) {

	// a few convenience variables
	struct Config* p = &transformer->config;
	struct Weights* w = &transformer->weights;
	struct RunState* s = &transformer->state;
	float* x = s->x;
	int dim = p->dim;
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
	int hidden_dim = p->hidden_dim;
	int head_size = dim / p->n_heads;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);
	assert(p->vocab_size % 32 == 0);

	// copy the token embedding into x
	assert(token < p->vocab_size);
	kernel_embed<<<dim / 32, 32>>>(x, (cudtype_t*)w->token_embedding_table + token * dim, dim);

	// forward all the layers
	for (unsigned long long l = 0; l < p->n_layers; l++) {

		// attention rmsnorm
		kernel_rmsnorm<<<1, 32>>>(s->xb, x, (cudtype_t*)w->rms_att_weight[l], dim);

		// key and value point to the kv cache
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
		s->k = s->key_cache + loff + pos * kv_dim;
		s->v = s->value_cache + loff + pos * kv_dim;

		// qkv matmuls for this position
		kernel_matmul_q<<<dim / 32, 32>>>(s->q, s->xb, (cudtype_t*)w->wq[l], dim, dim);
		kernel_matmul_kv<<<kv_dim / 32, 32>>>(s->k, s->xb, (cudtype_t*)w->wk[l], dim, kv_dim);
		kernel_matmul_kv<<<kv_dim / 32, 32>>>(s->v, s->xb, (cudtype_t*)w->wv[l], dim, kv_dim);

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		assert(dim % 64 == 0 && kv_dim % 64 == 0);
		kernel_rope<<<dim / 64, 32>>>(s->q, head_size, pos, p->rope_theta, dim);
		kernel_rope<<<kv_dim / 64, 32>>>(s->k, head_size, pos, p->rope_theta, kv_dim);

		// attention scores for all heads
		kernel_attn_score<<<dim3((pos + 1 + 31) / 32, p->n_heads), 32>>>(s->att, s->q, s->key_cache + loff, p->n_heads, head_size, p->seq_len, kv_dim, kv_mul, pos);

		// softmax the scores to get attention weights, from 0..pos inclusively
		kernel_attn_softmax<<<dim3(1, p->n_heads), 32>>>(s->att, p->n_heads, p->seq_len, pos);

		// compute weighted sum of the values into xb
		assert(head_size % 32 == 0);
		kernel_attn_mix<<<dim3(head_size / 32, p->n_heads), 32>>>(s->xb, s->att, s->value_cache + loff, p->n_heads, head_size, p->seq_len, kv_dim, kv_mul, pos);

		// final matmul to get the output of the attention
		kernel_matmul_attn<<<dim / 32, 32>>>(x, s->xb, (cudtype_t*)w->wo[l], dim, dim);

		// ffn rmsnorm
		kernel_rmsnorm<<<1, 32>>>(s->xb, x, (cudtype_t*)w->rms_ffn_weight[l], dim);

		// self.w2(F.silu(self.w1(x)) * self.w3(x)) + pre-rmsnorm residual
		kernel_matmul_ffn13<<<hidden_dim / 32, 32>>>(s->hb, s->xb, (cudtype_t*)w->w1[l], (cudtype_t*)w->w3[l], dim, hidden_dim);
		kernel_matmul_ffn2<<<dim / 32, 32>>>(x, s->hb, (cudtype_t*)w->w2[l], hidden_dim, dim);
	}

	// final rmsnorm
	kernel_rmsnorm<<<1, 32>>>(x, x, (cudtype_t*)w->rms_final_weight, dim);

	// classifier into logits
	kernel_matmul_cls<<<p->vocab_size / 32, 32>>>(s->logits, x, (cudtype_t*)w->wcls, p->dim, p->vocab_size);

	CUDA_SYNC();

	return s->logits;
}
