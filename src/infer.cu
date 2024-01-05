#include "model.h"

#include "profiler.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "helpers.cuh"

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

	cudaDeviceProp devprop = {};
	CUDA_CHECK(cudaGetDeviceProperties(&devprop, 0));

	printf("# CUDA: %s, compute %d.%d, %d SMs, %.1f GiB, peak bandwidth %.0f GB/s\n",
	       devprop.name, devprop.major, devprop.minor, devprop.multiProcessorCount,
	       (double)devprop.totalGlobalMem / (1024 * 1024 * 1024),
	       (double)devprop.memoryClockRate * (devprop.memoryBusWidth / 8) * 2 / 1e6);

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (float*)cuda_devicecopy(weights->rms_att_weight[l], dim * sizeof(float));
		weights->rms_ffn_weight[l] = (float*)cuda_devicecopy(weights->rms_ffn_weight[l], dim * sizeof(float));

		weights->wq[l] = cuda_devicecopy(weights->wq[l], dim * dim * weights->dsize);
		weights->wk[l] = cuda_devicecopy(weights->wk[l], dim * kv_dim * weights->dsize);
		weights->wv[l] = cuda_devicecopy(weights->wv[l], dim * kv_dim * weights->dsize);
		weights->wo[l] = cuda_devicecopy(weights->wo[l], dim * dim * weights->dsize);

		weights->w1[l] = cuda_devicecopy(weights->w1[l], dim * hidden_dim * weights->dsize);
		weights->w2[l] = cuda_devicecopy(weights->w2[l], dim * hidden_dim * weights->dsize);
		weights->w3[l] = cuda_devicecopy(weights->w3[l], dim * hidden_dim * weights->dsize);
	}

	weights->rms_final_weight = (float*)cuda_devicecopy(weights->rms_final_weight, dim * sizeof(float));
	weights->token_embedding_table = cuda_devicecopy(weights->token_embedding_table, config->vocab_size * dim * weights->dsize);
	weights->wcls = cuda_devicecopy(weights->wcls, dim * config->vocab_size * weights->dsize);

	state->x = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb = (float*)cuda_devicealloc(dim * sizeof(float));
	state->hb = (float*)cuda_devicealloc(hidden_dim * sizeof(float));
	state->q = (float*)cuda_devicealloc(dim * sizeof(float));
	state->k = (float*)cuda_devicealloc(kv_dim * sizeof(float));
	state->v = (float*)cuda_devicealloc(kv_dim * sizeof(float));
	state->att = (float*)cuda_devicealloc(config->n_heads * config->seq_len * sizeof(float));

	state->key_cache = (kvtype_t*)cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * sizeof(kvtype_t));
	state->value_cache = (kvtype_t*)cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * sizeof(kvtype_t));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)cuda_hostalloc(config->vocab_size * sizeof(float));
}

template <typename T>
__global__ static void kernel_embed(float* o, T* weight, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(i < size);

	o[i] = float(weight[i]);
}

__global__ static void kernel_rmsnorm(float* o, float* x, float* weight, int size) {
	int i = threadIdx.x;
	int blockSize = blockDim.x;

	extern __shared__ float xs[];

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j];
		ss += v * v;

		// premultiply x by weight into shared memory to accelerate the second loop
		xs[j] = v * weight[j];
	}

	// sum across threads in block
	ss = blockreduce_sum(ss);

	// compute scale
	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrtf(ss);

	// normalize and scale
	// note: blockreduce above implies __syncthreads so xs[] reads are safe
	for (int j = i; j < size; j += blockSize) {
		o[j] = xs[j] * ss;
	}
}

template <typename T>
__global__ static void kernel_matmul_cls(float* xout, float* x, T* w, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	// instead of writing one value per block, we transpose the values and write all results from first warp
	val = blocktranspose(val, 0.f);

	if (threadIdx.x < blockDim.x / warpSize) {
		xout[i + threadIdx.x] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_qkv(float* qout, float* kout, float* vout, float* x, T* wq, T* wk, T* wv, int n, int d, int kvd) {
	int i = blockIdx.x;
	assert(i < d + kvd * 2);

	float* out = i < d ? qout : (i < d + kvd ? kout : vout);
	T* w = i < d ? wq : (i < d + kvd ? wk : wv);
	int j = i < d ? i : (i < d + kvd ? i - d : i - d - kvd);

	float val = matmul_warppar(x, w, j, n, n);
	if (threadIdx.x == 0) {
		out[j] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_attn(float* xout, float* x, T* w, int n, int d) {
	int i = blockIdx.x;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	if (threadIdx.x == 0) {
		// += for residual
		xout[i] += val;
	}
}

template <typename T>
__global__ static void kernel_matmul_ffn13(float* xout, float* x, T* w1, T* w3, int n, int d) {
	int i = blockIdx.x;
	assert(i < d);

	float v1 = matmul_warppar(x, w1, i, n, n);
	float v3 = matmul_warppar(x, w3, i, n, n);

	// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	float val = v1;
	val *= 1.0f / (1.0f + expf(-v1));
	val *= v3;

	if (threadIdx.x == 0) {
		xout[i] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_ffn2(float* xout, float* x, T* w, int n, int d) {
	int i = blockIdx.x;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	if (threadIdx.x == 0) {
		// += for residual
		xout[i] += val;
	}
}

__global__ static void kernel_rope_qkv(float* q, float* k, float* v, kvtype_t* kb, kvtype_t* vb, int head_size, int pos, int kv_pos, int kv_sink, float theta_log2, int d, int kvd, int seq_len) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	assert(i < d + kvd + kv_sink * kvd);

	int j = i < d ? i : (i < d + kvd ? i - d : i - d - kvd);

	int head_dim = j & (head_size - 1);
	float freq = exp2f(-theta_log2 * (head_dim / (float)head_size));
	float fcr, fci;
	sincosf(pos * freq, &fci, &fcr);

	if (i < d) {
		float q0 = q[j];
		float q1 = q[j + 1];
		float rq0 = q0 * fcr - q1 * fci;
		float rq1 = q0 * fci + q1 * fcr;

		q[j] = rq0;
		q[j + 1] = rq1;
	} else if (i < d + kvd) {
		float k0 = k[j];
		float k1 = k[j + 1];
		float rk0 = k0 * fcr - k1 * fci;
		float rk1 = k0 * fci + k1 * fcr;

		float v0 = v[j];
		float v1 = v[j + 1];

		k[j] = rk0;
		k[j + 1] = rk1;

		// update kvcache key/value
		kb[kv_pos * kvd + j] = rk0;
		kb[kv_pos * kvd + j + 1] = rk1;

		// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
		vb[kv_pos + seq_len * j] = v0;
		vb[kv_pos + seq_len * (j + 1)] = v1;
	} else {
		// rotate sink tokens forward to keep pace with non-sink tokens
		float k0 = kb[j];
		float k1 = kb[j + 1];

		sincosf(freq, &fci, &fcr);

		float rk0 = k0 * fcr - k1 * fci;
		float rk1 = k0 * fci + k1 * fcr;

		kb[j] = rk0;
		kb[j + 1] = rk1;
	}
}

__global__ static void kernel_attn_score(float* attb, float* qb, kvtype_t* kb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int t = blockIdx.x;
	assert(t < kv_len);

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* q = qb + h * head_size;
	kvtype_t* k = kb + t * kv_dim + kvh * head_size;
	float* att = attb + h * seq_len;

	float score = 0.0f;
	for (int j = threadIdx.x * 2; j < head_size; j += warpSize * 2) {
		float2 kk = __half22float2(*((half2*)&k[j]));
		float2 qq = *(float2*)&q[j];
		score += kk.x * qq.x;
		score += kk.y * qq.y;
	}

	score = warpreduce_sum(score);
	score /= sqrtf(head_size);

	if (threadIdx.x == 0) {
		att[t] = score;
	}
}

__global__ static void kernel_attn_softmax(float* attb, int n_heads, int seq_len, int kv_len) {
	int i = threadIdx.x;

	int h = blockIdx.x;
	assert(h < n_heads);

	float* att = attb + h * seq_len;

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < kv_len; j += blockDim.x) {
		max_val = max(max_val, att[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(max_val);

	// exp and sum per thread
	float sum = 0.0f;
	for (int j = i; j < kv_len; j += blockDim.x) {
		sum += expf(att[j] - max_val);
	}

	// sum across threads in block
	sum = blockreduce_sum(sum);

	// output normalized values
	for (int j = i; j < kv_len; j += blockDim.x) {
		att[j] = expf(att[j] - max_val) / sum;
	}
}

__global__ static void kernel_attn_mix(float* xout, float* attb, kvtype_t* valb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int i = blockIdx.x;
	assert(i < head_size);

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* att = attb + h * seq_len;
	kvtype_t* val = valb + (kvh * head_size + i) * seq_len;

	float res = 0.0f;
	for (int t = threadIdx.x * 2; t + 1 < kv_len; t += warpSize * 2) {
		float2 vv = __half22float2(*((half2*)&val[t]));
		float2 aa = *(float2*)&att[t];
		res += vv.x * aa.x;
		res += vv.y * aa.y;
	}

	if (kv_len % 2 == 1 && threadIdx.x == 0) {
		res += att[kv_len - 1] * float(val[kv_len - 1]);
	}

	res = warpreduce_sum(res);

	if (threadIdx.x == 0) {
		xout[h * head_size + i] = res;
	}
}

template <typename T>
static float* forward(struct Transformer* transformer, int token, int pos, unsigned flags) {
	profiler_begin();

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

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);
	assert(p->vocab_size % 32 == 0);

	// rmsnorm and softmax require a larger-than-warp block size for efficiency
	const int rmsnorm_size = 1024;
	const int softmax_size = 1024;

	// copy the token embedding into x
	assert(token < p->vocab_size);
	kernel_embed<<<dim / 32, 32>>>(x, (T*)w->token_embedding_table + token * dim, dim);
	profiler_trigger("embed", 0);

	// forward all the layers
	for (int l = 0; l < p->n_layers; l++) {
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		// attention rmsnorm
		kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float)>>>(s->xb, x, w->rms_att_weight[l], dim);
		profiler_trigger("rmsnorm", 0);

		// qkv matmuls for this position
		kernel_matmul_qkv<<<dim + kv_dim * 2, 32>>>(s->q, s->k, s->v, s->xb, (T*)w->wq[l], (T*)w->wk[l], (T*)w->wv[l], dim, dim, kv_dim);
		profiler_trigger("matmul_qkv", (dim + kv_dim * 2) * dim * sizeof(T));

		// RoPE relative positional encoding: complex-valued rotate q and k in each head, and update kv cache
		assert(dim % 64 == 0 && kv_dim % 64 == 0);
		assert((head_size & (head_size - 1)) == 0); // head_size must be a power of 2
		kernel_rope_qkv<<<(dim + kv_dim + kv_dim * kv_sink) / 64, 32>>>(s->q, s->k, s->v, s->key_cache + loff, s->value_cache + loff, head_size, pos, kv_pos, kv_sink, log2(p->rope_theta), dim, kv_dim, p->seq_len);
		profiler_trigger("rope_qkv", 0);

		// only update kv cache and don't output logits
		if (l == p->n_layers - 1 && (flags & FF_UPDATE_KV_ONLY) != 0) {
			break;
		}

		// attention scores for all heads
		kernel_attn_score<<<dim3(kv_len, p->n_kv_heads), dim3(32, kv_mul)>>>(s->att, s->q, s->key_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);
		profiler_trigger("attn_score", p->n_kv_heads * kv_len * head_size * sizeof(kvtype_t));

		// softmax the scores to get attention weights over [0..kv_len)
		kernel_attn_softmax<<<p->n_heads, softmax_size>>>(s->att, p->n_heads, p->seq_len, kv_len);
		profiler_trigger("attn_softmax", 0);

		// compute weighted sum of the values into xb
		assert(head_size % 32 == 0);
		kernel_attn_mix<<<dim3(head_size, p->n_kv_heads), dim3(32, kv_mul)>>>(s->xb, s->att, s->value_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);
		profiler_trigger("attn_mix", p->n_kv_heads * kv_len * head_size * sizeof(kvtype_t));

		// final matmul to get the output of the attention
		kernel_matmul_attn<<<dim, 32>>>(x, s->xb, (T*)w->wo[l], dim, dim);
		profiler_trigger("matmul_attn", dim * dim * sizeof(T));

		// ffn rmsnorm
		kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float)>>>(s->xb, x, w->rms_ffn_weight[l], dim);
		profiler_trigger("rmsnorm", 0);

		// self.w2(F.silu(self.w1(x)) * self.w3(x)) + pre-rmsnorm residual
		kernel_matmul_ffn13<<<hidden_dim, 32>>>(s->hb, s->xb, (T*)w->w1[l], (T*)w->w3[l], dim, hidden_dim);
		profiler_trigger("matmul_ffn13", 2 * hidden_dim * dim * sizeof(T));

		kernel_matmul_ffn2<<<dim, 32>>>(x, s->hb, (T*)w->w2[l], hidden_dim, dim);
		profiler_trigger("matmul_ffn2", dim * hidden_dim * sizeof(T));
	}

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		profiler_endsync();

		return NULL;
	}

	// final rmsnorm
	kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float)>>>(x, x, w->rms_final_weight, dim);
	profiler_trigger("rmsnorm", 0);

	// classifier into logits
	kernel_matmul_cls<<<p->vocab_size / 32, 32 * 32>>>(s->logits, x, (T*)w->wcls, dim, p->vocab_size);
	profiler_trigger("matmul_cls", p->vocab_size * dim * sizeof(T));

	profiler_endsync();

	CUDA_SYNC();

	return s->logits;
}

extern "C" float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags) {
	switch (transformer->weights.dsize) {
	case 1:
		return forward<__nv_fp8_e5m2>(transformer, token, pos, flags);
	case 2:
		return forward<half>(transformer, token, pos, flags);
	default:
		return NULL;
	}
}
