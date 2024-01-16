#include "model.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
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

#define PROF_TOKEN(bytes) ((0xCDAFull << 48) | (bytes))

static cudaStream_t stream, parstream;
static cudaEvent_t parsync[2];

static void* cuda_devicecopy(void* host, size_t size) {
	if (host == NULL) {
		return NULL;
	}
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

	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaStreamCreate(&parstream));

	for (int i = 0; i < sizeof(parsync) / sizeof(parsync[0]); ++i) {
		CUDA_CHECK(cudaEventCreateWithFlags(&parsync[i], cudaEventDisableTiming));
	}

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
	size_t dbits = weights->dbits; // size_t prevents integer overflow in multiplications below

	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (float*)cuda_devicecopy(weights->rms_att_weight[l], dim * sizeof(float));
		weights->rms_ffn_weight[l] = (float*)cuda_devicecopy(weights->rms_ffn_weight[l], dim * sizeof(float));
		weights->ln_weight[l] = (float*)cuda_devicecopy(weights->ln_weight[l], dim * sizeof(float));
		weights->ln_bias[l] = (float*)cuda_devicecopy(weights->ln_bias[l], dim * sizeof(float));

		weights->wq[l] = cuda_devicecopy(weights->wq[l], dim * dim * dbits / 8);
		weights->wk[l] = cuda_devicecopy(weights->wk[l], dim * kv_dim * dbits / 8);
		weights->wv[l] = cuda_devicecopy(weights->wv[l], dim * kv_dim * dbits / 8);
		weights->wo[l] = cuda_devicecopy(weights->wo[l], dim * dim * dbits / 8);

		weights->w1[l] = cuda_devicecopy(weights->w1[l], dim * hidden_dim * dbits / 8);
		weights->w2[l] = cuda_devicecopy(weights->w2[l], dim * hidden_dim * dbits / 8);
		weights->w3[l] = cuda_devicecopy(weights->w3[l], dim * hidden_dim * dbits / 8);

		weights->bq[l] = (float*)cuda_devicecopy(weights->bq[l], dim * sizeof(float));
		weights->bk[l] = (float*)cuda_devicecopy(weights->bk[l], kv_dim * sizeof(float));
		weights->bv[l] = (float*)cuda_devicecopy(weights->bv[l], kv_dim * sizeof(float));
		weights->bo[l] = (float*)cuda_devicecopy(weights->bo[l], dim * sizeof(float));

		weights->b1[l] = (float*)cuda_devicecopy(weights->b1[l], hidden_dim * sizeof(float));
		weights->b2[l] = (float*)cuda_devicecopy(weights->b2[l], dim * sizeof(float));
	}

	weights->rms_final_weight = (float*)cuda_devicecopy(weights->rms_final_weight, dim * sizeof(float));
	weights->ln_final_weight = (float*)cuda_devicecopy(weights->ln_final_weight, dim * sizeof(float));
	weights->ln_final_bias = (float*)cuda_devicecopy(weights->ln_final_bias, dim * sizeof(float));
	weights->token_embedding_table = cuda_devicecopy(weights->token_embedding_table, config->vocab_size * dim * (size_t)dbits / 8);
	weights->wcls = cuda_devicecopy(weights->wcls, dim * config->vocab_size * (size_t)dbits / 8);
	weights->bcls = (float*)cuda_devicecopy(weights->bcls, config->vocab_size * sizeof(float));

	state->x = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb2 = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xa = (float*)cuda_devicealloc(dim * sizeof(float));
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
__global__ static void kernel_embed(float* o, T* weight, int token, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(i < n);

	o[i] = float(weight[token * n + i]);
}

__global__ static void kernel_embed(float* o, uint32_t* weight, int token, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(i < n);

	uint32_t wg = weight[token * n / 8 + i / 8];
	float wgs = -fp8_e5m2_ff(wg & 0xff) / 4.f;

	o[i] = (int((wg >> (8 + (i % 8) * 3)) & 7) - 4) * wgs;
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

	// normalize and scale
	// note: blockreduce above implies __syncthreads so xs[] reads are safe
	float scale = rsqrtf(ss / size + 1e-5f);
	for (int j = i; j < size; j += blockSize) {
		o[j] = xs[j] * scale;
	}
}

__global__ static void kernel_layernorm(float* o, float* x, float* acc, float* weight, float* bias, int size) {
	int i = threadIdx.x;
	int blockSize = blockDim.x;

	float K = x[0] + (acc ? acc[0] : 0.f); // shifted variance for numerical stability
	__syncthreads();

	// calculate sum and sum of squares (per thread)
	float sum = 0.0f, ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j];
		if (acc) {
			v += acc[j];
			x[j] = v;
		}

		sum += v - K;
		ss += (v - K) * (v - K);
	}

	// sum across threads in block
	sum = blockreduce_sum(sum);
	ss = blockreduce_sum(ss);

	float rsize = 1.f / size;
	float mean = sum * rsize + K;
	float var = (ss - sum * sum * rsize) * rsize;

	// normalize and scale
	float scale = rsqrtf(var + 1e-5f);
	for (int j = i; j < size; j += blockSize) {
		o[j] = (x[j] - mean) * scale * weight[j] + bias[j];
	}
}

template <typename T>
__global__ static void kernel_matmul_cls(uint64_t, float* xout, float* x, T* w, float* b, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	if (b) {
		val += b[i];
	}

	// instead of writing one value per block, we transpose the values and write all results from first warp
	val = blocktranspose(val, 0.f);

	if (threadIdx.x < blockDim.x / warpSize) {
		xout[i + threadIdx.x] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_qkv(uint64_t, float* qout, float* kout, float* vout, float* x, T* wq, T* wk, T* wv, float* bq, float* bk, float* bv, int n, int d, int kvd) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d + kvd * 2);

	float* out = i < d ? qout : (i < d + kvd ? kout : vout);
	T* w = i < d ? wq : (i < d + kvd ? wk : wv);
	float* b = i < d ? bq : (i < d + kvd ? bk : bv);
	int j = i < d ? i : (i < d + kvd ? i - d : i - d - kvd);

	float val = matmul_warppar(x, w, j, n, n);

	if (b) {
		val += b[j];
	}

	if (threadIdx.x % warpSize == 0) {
		out[j] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_attn(uint64_t, float* xout, float* x, T* w, float* b, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	if (b) {
		val += b[i];
	}

	if (threadIdx.x % warpSize == 0) {
		// += for residual
		xout[i] += val;
	}
}

template <typename T>
__global__ static void kernel_matmul_ffn13_silu(uint64_t, float* xout, float* x, T* w1, T* w3, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float v1 = matmul_warppar(x, w1, i, n, n);
	float v3 = matmul_warppar(x, w3, i, n, n);

	// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	float val = v1;
	val *= 1.0f / (1.0f + expf(-v1));
	val *= v3;

	if (threadIdx.x % warpSize == 0) {
		xout[i] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_ffn1_gelu(uint64_t, float* xout, float* x, T* w1, float* b1, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float v1 = matmul_warppar(x, w1, i, n, n);

	float val = v1 + b1[i];

	// GELU (0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))))
	val = 0.5f * val * (1.0f + tanhf(0.797885f * (val + 0.044715f * val * val * val)));

	if (threadIdx.x % warpSize == 0) {
		xout[i] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_ffn2(uint64_t, float* xout, float* x, T* w, float* acc, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

	if (threadIdx.x % warpSize == 0) {
		xout[i] = val + acc[i];
	}
}

__global__ static void kernel_rope_qkv(float* q, float* k, float* v, kvtype_t* kb, kvtype_t* vb, int head_size, int pos, int kv_pos, int kv_sink, float theta_log2, int d, int kvd, int seq_len, int rotary_dim) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	assert(i < d + kvd + kv_sink * kvd);

	int j = i < d ? i : (i < d + kvd ? i - d : i - d - kvd);

	int j_head = j % head_size; // TODO: optimize when head_size is a power of two
	float freq = j_head >= rotary_dim ? 0.f : exp2f(-theta_log2 * (float)j_head / (float)rotary_dim);
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

		// update kvcache key/value
		// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
		// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
		kb[kv_pos * 2 + 0 + seq_len * j] = rk0;
		kb[kv_pos * 2 + 1 + seq_len * j] = rk1;
		vb[kv_pos + seq_len * (j + 0)] = v0;
		vb[kv_pos + seq_len * (j + 1)] = v1;
	} else {
		// rotate sink tokens forward to keep pace with non-sink tokens
		// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
		int t = j / kvd;
		int o = t * 2 + seq_len * (j % kvd);

		float k0 = kb[o];
		float k1 = kb[o + 1];

		sincosf(freq, &fci, &fcr);

		float rk0 = k0 * fcr - k1 * fci;
		float rk1 = k0 * fci + k1 * fcr;

		kb[o] = rk0;
		kb[o + 1] = rk1;
	}
}

__global__ static void kernel_attn_score(uint64_t, float* attb, float* qb, kvtype_t* kb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= kv_len) {
		return;
	}

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* qh = qb + h * head_size;
	kvtype_t* kh = kb + kvh * head_size * seq_len;
	float* atth = attb + h * seq_len;

	float score = 0.0f;
	for (int j = 0; j < head_size; j += 2) {
		float2 kk = __half22float2(*((half2*)&kh[j * seq_len + t * 2]));
		float2 qq = *(float2*)&qh[j];
		score += kk.x * qq.x;
		score += kk.y * qq.y;
	}

	score /= sqrtf(head_size);

	atth[t] = score;
}

__global__ static void kernel_attn_softmax(float* attb, int n_heads, int seq_len, int kv_len) {
	int i = threadIdx.x;

	int h = blockIdx.x;
	assert(h < n_heads);

	float* atth = attb + h * seq_len;

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < kv_len; j += blockDim.x) {
		max_val = max(max_val, atth[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(max_val);

	// exp and sum per thread
	float sum = 0.0f;
	for (int j = i; j < kv_len; j += blockDim.x) {
		sum += expf(atth[j] - max_val);
	}

	// sum across threads in block
	sum = blockreduce_sum(sum);

	// output normalized values
	for (int j = i; j < kv_len; j += blockDim.x) {
		atth[j] = expf(atth[j] - max_val) / sum;
	}
}

__global__ static void kernel_attn_mix(uint64_t, float* xout, float* attb, kvtype_t* valb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int i = blockIdx.x;
	assert(i < head_size);

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* atth = attb + h * seq_len;
	kvtype_t* vh = valb + kvh * head_size * seq_len;
	kvtype_t* val = vh + i * seq_len;

	float res = 0.0f;
	for (int t = threadIdx.x * 2; t < kv_len - 1; t += warpSize * 2) {
		float2 vv = __half22float2(*((half2*)&val[t]));
		float2 aa = *(float2*)&atth[t];
		res += vv.x * aa.x;
		res += vv.y * aa.y;
	}

	if (kv_len % 2 == 1 && threadIdx.x == 0) {
		res += atth[kv_len - 1] * float(val[kv_len - 1]);
	}

	res = warpreduce_sum(res);

	if (threadIdx.x == 0) {
		xout[h * head_size + i] = res;
	}
}

template <typename T>
static float* forward(struct Transformer* transformer, int token, int pos, unsigned flags) {
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
	size_t dbits = w->dbits; // size_t prevents integer overflow in multiplications below

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;

	// ensure all dimensions are warp-aligned
	assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);
	assert(p->vocab_size % 32 == 0);

	// rmsnorm and softmax require a larger-than-warp block size for efficiency
	const int layernorm_size = 1024;
	const int rmsnorm_size = 1024;
	const int softmax_size = p->arch == Phi ? 512 : 1024;

	// gf4 kernels need a little more parallelism to saturate the GPU
	const int matmul_par = w->dbits == 4 ? 2 : 1;

	// copy the token embedding into x
	assert(token < p->vocab_size);
	kernel_embed<<<dim / 32, 32, 0, stream>>>(x, (T*)w->token_embedding_table, token, dim);

	// forward all the layers
	for (int l = 0; l < p->n_layers; l++) {
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		if (p->arch == Phi) {
			// input layernorm
			kernel_layernorm<<<1, layernorm_size, 0, stream>>>(s->xb, x, l == 0 ? NULL : s->xa, w->ln_weight[l], w->ln_bias[l], dim);

			if (parstream) {
				// wait for layernorm to complete on parstream
				CUDA_CHECK(cudaEventRecord(parsync[0], stream));
				CUDA_CHECK(cudaStreamWaitEvent(parstream, parsync[0], 0));
			}
		} else {
			// attention rmsnorm
			kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(s->xb, x, w->rms_att_weight[l], dim);
		}

		// qkv matmuls for this position
		kernel_matmul_qkv<<<(dim + kv_dim * 2) / matmul_par, 32 * matmul_par, 0, stream>>>(
		    PROF_TOKEN((dim + kv_dim * 2) * dim * dbits / 8), s->q, s->k, s->v, s->xb, (T*)w->wq[l], (T*)w->wk[l], (T*)w->wv[l], w->bq[l], w->bk[l], w->bv[l], dim, dim, kv_dim);

		// RoPE relative positional encoding: complex-valued rotate q and k in each head, and update kv cache
		assert(dim % 64 == 0 && kv_dim % 64 == 0);
		kernel_rope_qkv<<<(dim + kv_dim + kv_dim * kv_sink) / 64, 32, 0, stream>>>(s->q, s->k, s->v, s->key_cache + loff, s->value_cache + loff, head_size, pos, kv_pos, kv_sink, log2(p->rope_theta), dim, kv_dim, p->seq_len, p->rotary_dim);

		// only update kv cache and don't output logits
		if (l == p->n_layers - 1 && (flags & FF_UPDATE_KV_ONLY) != 0) {
			break;
		}

		size_t kvbw = p->n_kv_heads * head_size * kv_len * sizeof(kvtype_t);

		// attention scores for all heads
		kernel_attn_score<<<dim3((kv_len + 31) / 32, p->n_kv_heads), dim3(32, kv_mul), 0, stream>>>(
		    PROF_TOKEN(kvbw), s->att, s->q, s->key_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);

		// softmax the scores to get attention weights over [0..kv_len)
		kernel_attn_softmax<<<p->n_heads, softmax_size, 0, stream>>>(s->att, p->n_heads, p->seq_len, kv_len);

		// compute weighted sum of the values into xb2
		kernel_attn_mix<<<dim3(head_size, p->n_kv_heads), dim3(32, kv_mul), 0, stream>>>(
		    PROF_TOKEN(kvbw), s->xb2, s->att, s->value_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);

		// final matmul to get the output of the attention
		kernel_matmul_attn<<<dim / matmul_par, 32 * matmul_par, 0, stream>>>(
		    PROF_TOKEN(dim * dim * dbits / 8), x, s->xb2, (T*)w->wo[l], w->bo[l], dim, dim);

		if (p->arch == Phi) {
			cudaStream_t mlpstream = parstream ? parstream : stream;

			// self.w2(F.gelu(self.w1(x))) + pre-rmsnorm residual
			kernel_matmul_ffn1_gelu<<<hidden_dim / matmul_par, 32 * matmul_par, 0, mlpstream>>>(
			    PROF_TOKEN(hidden_dim * dim * dbits / 8), s->hb, s->xb, (T*)w->w1[l], w->b1[l], dim, hidden_dim);

			kernel_matmul_ffn2<<<dim / matmul_par, 32 * matmul_par, 0, mlpstream>>>(
			    PROF_TOKEN(dim * hidden_dim * dbits / 8), s->xa, s->hb, (T*)w->w2[l], w->b2[l], hidden_dim, dim);

			if (parstream) {
				// MLP atomically aggregates results into x[] which is used by main stream on next iteration, so wait for that
				CUDA_CHECK(cudaEventRecord(parsync[1], parstream));
				CUDA_CHECK(cudaStreamWaitEvent(stream, parsync[1], 0));
			}
		} else {
			// ffn rmsnorm
			kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(s->xb, x, w->rms_ffn_weight[l], dim);

			// self.w2(F.silu(self.w1(x)) * self.w3(x)) + pre-rmsnorm residual
			kernel_matmul_ffn13_silu<<<hidden_dim / matmul_par, 32 * matmul_par, 0, stream>>>(
			    PROF_TOKEN(2 * hidden_dim * dim * dbits / 8), s->hb, s->xb, (T*)w->w1[l], (T*)w->w3[l], dim, hidden_dim);

			kernel_matmul_ffn2<<<dim / matmul_par, 32 * matmul_par, 0, stream>>>(
			    PROF_TOKEN(dim * hidden_dim * dbits / 8), x, s->hb, (T*)w->w2[l], x, hidden_dim, dim);
		}
	}

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	if (p->arch == Phi) {
		// final layernorm
		kernel_layernorm<<<1, layernorm_size, 0, stream>>>(x, x, s->xa, w->ln_final_weight, w->ln_final_bias, dim);
	} else {
		// final rmsnorm
		kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(x, x, w->rms_final_weight, dim);
	}

	// classifier into logits
	kernel_matmul_cls<<<p->vocab_size / 32, 32 * 32, 0, stream>>>(
	    PROF_TOKEN(p->vocab_size * dim * dbits / 8), s->logits, x, (T*)w->wcls, w->bcls, dim, p->vocab_size);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	return s->logits;
}

extern "C" float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags) {
	switch (transformer->weights.dbits) {
	case 4:
		return forward<uint32_t>(transformer, token, pos, flags);
	case 8:
		return forward<__nv_fp8_e5m2>(transformer, token, pos, flags);
	case 16:
		return forward<half>(transformer, token, pos, flags);
	default:
		assert(!"Unsupported dbits: must be 4, 8 or 16 for CUDA");
		return NULL;
	}
}
