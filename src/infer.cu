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
	void* device = NULL;
	CUDA_CHECK(cudaMalloc(&device, size));
	CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
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

extern "C" void* upload_cuda(void* host, size_t size) {
	return cuda_devicecopy(host, size);
}

extern "C" void prepare_cuda(struct Transformer* transformer) {
	struct Config* config = &transformer->config;
	struct Weights* weights = &transformer->weights;
	struct RunState* state = &transformer->state;

	cudaDeviceProp devprop = {};
	CUDA_CHECK(cudaGetDeviceProperties(&devprop, 0));

	printf("# CUDA: %s, compute %d.%d, %d SMs, %.1f GiB, peak bandwidth %.0f GB/s (ECC %d)\n",
	       devprop.name, devprop.major, devprop.minor, devprop.multiProcessorCount,
	       (double)devprop.totalGlobalMem / (1024 * 1024 * 1024),
	       (double)devprop.memoryClockRate * (devprop.memoryBusWidth / 8) * 2 / 1e6, devprop.ECCEnabled);

	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaStreamCreate(&parstream));

	for (int i = 0; i < sizeof(parsync) / sizeof(parsync[0]); ++i) {
		CUDA_CHECK(cudaEventCreateWithFlags(&parsync[i], cudaEventDisableTiming));
	}

	int dim = config->dim;
	int hidden_dim = config->hidden_dim;
	int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

	if (config->n_experts) {
		for (int l = 0; l < config->n_layers; ++l) {
			weights->moewr[l][0] = (void**)cuda_devicecopy(weights->moew1[l], config->n_experts * sizeof(void*));
			weights->moewr[l][1] = (void**)cuda_devicecopy(weights->moew2[l], config->n_experts * sizeof(void*));
			weights->moewr[l][2] = (void**)cuda_devicecopy(weights->moew3[l], config->n_experts * sizeof(void*));
		}
	}

	state->x = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xb2 = (float*)cuda_devicealloc(dim * sizeof(float));
	state->xa = (float*)cuda_devicealloc(dim * sizeof(float));
	state->hb = (float*)cuda_devicealloc(hidden_dim * sizeof(float));
	state->he = (float*)cuda_devicealloc(config->n_experts_ac * hidden_dim * sizeof(float));
	state->q = (float*)cuda_devicealloc(dim * sizeof(float));
	state->att = (float*)cuda_devicealloc(config->n_heads * config->seq_len * sizeof(float));
	state->exp = (float*)cuda_devicealloc((config->n_experts + config->n_experts_ac * 2) * sizeof(float));

	assert(state->kvbits == 8 || state->kvbits == 16);
	state->key_cache = cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));
	state->value_cache = cuda_devicealloc((size_t)config->n_layers * config->seq_len * kv_dim * (state->kvbits / 8));

	// logits are going to be read by the host so we just allocate them in host and write to host directly
	state->logits = (float*)cuda_hostalloc(config->vocab_size * sizeof(float));
}

template <typename T>
__device__ inline float embed(T* weight, int idx) {
	return float(weight[idx]);
}

__device__ inline float embed(uint32_t* weight, int idx) {
	return gf4_ff(weight[idx / 8], idx % 8);
}

template <typename T>
__global__ static void kernel_embed(float* o, T* weight, int token, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	assert(i < n);

	o[i] = embed(weight, token * n + i);
}

__global__ static void kernel_rmsnorm(float* o, float* x, float* weight, int size, float eps) {
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
	float scale = rsqrtf(ss / size + eps);
	for (int j = i; j < size; j += blockSize) {
		o[j] = xs[j] * scale;
	}
}

__global__ static void kernel_layernorm(float* o, float* x, float* acc, float* weight, int size, float eps) {
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
	float scale = rsqrtf(var + eps);
	for (int j = i; j < size; j += blockSize) {
		o[j] = (x[j] - mean) * scale * weight[j];
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

template <typename T, typename KVT>
__global__ static void kernel_matmul_rope_qkv(uint64_t, float* qout, float* x, T* wq, T* wk, T* wv, float* bq, float* bk, float* bv, int n, int d, int kvd,
                                              KVT* kb, KVT* vb, int head_size, int pos, int kv_pos, int kv_sink, float theta_log2, int seq_len, int rotary_dim) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d + kvd * 2 + kv_sink * kvd);

	int j_head = i % head_size; // TODO: optimize when head_size is a power of two
	float freq = j_head >= rotary_dim ? 0.f : exp2f(-theta_log2 * (float)j_head / (float)rotary_dim);
	float fcr, fci;
	sincosf(pos * freq, &fci, &fcr);

	if (i >= d + kvd * 2) {
		int j = i - (d + kvd * 2);

		// rotate sink tokens forward to keep pace with non-sink tokens
		// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
		int t = j / kvd;
		int o = t * 2 + seq_len * (j % kvd);

		float k0 = float(kb[o]);
		float k1 = float(kb[o + 1]);

		sincosf(freq, &fci, &fcr);

		float rk0 = k0 * fcr - k1 * fci;
		float rk1 = k0 * fci + k1 * fcr;

		kb[o] = KVT(rk0);
		kb[o + 1] = KVT(rk1);
		return;
	}

	T* w = i < d ? wq : (i < d + kvd ? wk : wv);
	float* b = i < d ? bq : (i < d + kvd ? bk : bv);
	int j = i < d ? i : (i < d + kvd ? i - d : i - d - kvd);

	float val = matmul_warppar(x, w, j, n, n);

	if (b) {
		val += b[j];
	}

	__shared__ float vs[2];
	vs[threadIdx.x / warpSize] = val;
	__syncthreads();

	if (threadIdx.x == 0) {
		if (i < d) {
			// q
			float q0 = vs[0];
			float q1 = vs[1];
			float rq0 = q0 * fcr - q1 * fci;
			float rq1 = q0 * fci + q1 * fcr;

			qout[j] = rq0;
			qout[j + 1] = rq1;
		} else if (i < d + kvd) {
			// k
			float k0 = vs[0];
			float k1 = vs[1];
			float rk0 = k0 * fcr - k1 * fci;
			float rk1 = k0 * fci + k1 * fcr;

			// note: k layout is transposed (we store all positions for a given head *pair* contiguously) to improve attn_score performance
			kb[kv_pos * 2 + 0 + seq_len * j] = KVT(rk0);
			kb[kv_pos * 2 + 1 + seq_len * j] = KVT(rk1);
		} else {
			// v
			float v0 = vs[0];
			float v1 = vs[1];

			// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
			vb[kv_pos + seq_len * (j + 0)] = KVT(v0);
			vb[kv_pos + seq_len * (j + 1)] = KVT(v1);
		}
	}
}

template <typename T>
__global__ static void kernel_matmul_attn(uint64_t, float* xout, float* x, T* w, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	float val = matmul_warppar(x, w, i, n, n);

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
	float val = (v1 / (1.0f + expf(-v1))) * v3;

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

template <typename T>
__global__ static void kernel_moe_gate(float* moe_weights, int* moe_experts, float* x, T* w, int n, int experts, int active) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < experts);

	float val = matmul_warppar(x, w, i, n, n);

	__shared__ float ws[32];
	ws[i] = val;
	__syncthreads();

	// warp-parallel top expert selection within the first warp
	if (threadIdx.x < warpSize) {
		i = threadIdx.x;

		// (unscaled) softmax across experts
		float w = (i < experts) ? ws[i] : -FLT_MAX;
		float max_val = warpreduce_max(w);
		w = expf(w - max_val);

		// weight in top 24 bits, index in bottom 8
		int wi = (*(int*)&w & 0xffffff00) | i;

		// top k within warp
		float sumw = 0.f;
		int acti = -1;

		for (int k = 0; k < active; ++k) {
			int maxi = warpreduce_maxi(wi);

			sumw += *(float*)&maxi;

			// keeps top weight in thread k, clears weight for thread with max thread to avoid re-selection
			acti = (i == k) ? maxi : acti;
			wi = (wi == maxi) ? 0 : wi;
		}

		// write normalized weights
		if (i < active) {
			assert(acti >= 0);

			moe_experts[i] = acti & 0xff;
			moe_weights[i] = *(float*)&acti / sumw;
		}
	}
}

template <typename T>
__global__ static void kernel_matmul_moe_ffn13_silu(uint64_t, float* xout, float* x, T** w1, T** w3, int* moe_experts, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	int e = threadIdx.y;

	float v1 = matmul_warppar(x, w1[moe_experts[e]], i, n, n);
	float v3 = matmul_warppar(x, w3[moe_experts[e]], i, n, n);

	// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
	float val = (v1 / (1.0f + expf(-v1))) * v3;

	if (threadIdx.x % warpSize == 0) {
		xout[i + e * d] = val;
	}
}

template <typename T>
__global__ static void kernel_matmul_moe_ffn2(uint64_t, float* xout, float* x, T** w, int* moe_experts, float* moe_weights, int n, int d) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < d);

	int e = threadIdx.y;

	float val = matmul_warppar(x + e * n, w[moe_experts[e]], i, n, n);

	__shared__ float rs[32];
	rs[threadIdx.y] = val;
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float acc = 0.f;
		for (int k = 0; k < blockDim.y; ++k) {
			acc += rs[k] * moe_weights[k];
		}
		xout[i] += acc;
	}
}

union half4 {
	float2 g;
	half h[4];
};

__device__ inline float4 attn_load4(half* p) {
	half4 h = *(half4*)p;
	return {__half2float(h.h[0]), __half2float(h.h[1]), __half2float(h.h[2]), __half2float(h.h[3])};
}

__device__ inline float4 attn_load4(__nv_fp8_e5m2* p) {
	return fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)p);
}

template <typename KVT>
__global__ static void kernel_attn_score(uint64_t, float* attb, float* qb, KVT* kb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int t = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	if (t >= kv_len) {
		return;
	}

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* qh = qb + h * head_size;
	KVT* kh = kb + kvh * head_size * seq_len;
	float* atth = attb + h * seq_len;

	float score1 = 0.0f;
	float score2 = 0.0f;
	for (int j = 0; j < head_size; j += 2) {
		float4 kk = attn_load4(&kh[j * seq_len + t * 2]);
		float2 qq = *(float2*)&qh[j];
		score1 += kk.x * qq.x;
		score1 += kk.y * qq.y;
		score2 += kk.z * qq.x;
		score2 += kk.w * qq.y;
	}

	score1 /= sqrtf(head_size);
	score2 /= sqrtf(head_size);

	atth[t + 0] = score1;
	atth[t + 1] = score2;
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

	// exp per thread
	for (int j = i; j < kv_len; j += blockDim.x) {
		atth[j] = expf(atth[j] - max_val);
	}
}

template <typename KVT>
__global__ static void kernel_attn_mix(uint64_t, float* xout, float* attb, KVT* valb, int n_kv_heads, int head_size, int seq_len, int kv_dim, int kv_mul, int kv_len) {
	int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	assert(i < head_size);

	int kvh = blockIdx.y;
	assert(kvh < n_kv_heads);

	int h = kvh * kv_mul + threadIdx.y;

	float* atth = attb + h * seq_len;
	KVT* vh = valb + kvh * head_size * seq_len;
	KVT* val = vh + i * seq_len;

	int kv_len4 = kv_len & ~3;
	int lane = threadIdx.x % warpSize;

	float res = 0.0f;
	float sum = 0.0f;
	for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
		float4 vv = attn_load4(&val[t]);
		float4 aa = *(float4*)&atth[t];
		res += vv.x * aa.x;
		res += vv.y * aa.y;
		res += vv.z * aa.z;
		res += vv.w * aa.w;
		sum += aa.x + aa.y + aa.z + aa.w;
	}

	if (kv_len4 + lane < kv_len) {
		float a = atth[kv_len4 + lane];
		res += a * float(val[kv_len4 + lane]);
		sum += a;
	}

	res = warpreduce_sum(res);
	sum = warpreduce_sum(sum);

	if (threadIdx.x % warpSize == 0) {
		xout[h * head_size + i] = res / sum;
	}
}

template <typename T, typename KVT>
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

	// gf4 kernels need a little more parallelism to saturate 4090
	// fp8/fp16 kernels work well with 1 on 4090, but A100/H100 benefits from 2
	const int matmul_par = 2;

	// A100/H100 need higher parallelism for attention kernels
	const int attn_par = 2;

	// copy the token embedding into x
	assert(token < p->vocab_size);
	kernel_embed<<<dim / 32, 32, 0, stream>>>(x, (T*)w->token_embedding_table, token, dim);

	// forward all the layers
	for (int l = 0; l < p->n_layers; l++) {
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		if (p->arch == Phi) {
			// input layernorm
			kernel_layernorm<<<1, layernorm_size, 0, stream>>>(s->xb, x, l == 0 ? NULL : s->xa, w->ln_weight[l], dim, p->norm_eps);

			if (parstream) {
				// wait for layernorm to complete on parstream
				CUDA_CHECK(cudaEventRecord(parsync[0], stream));
				CUDA_CHECK(cudaStreamWaitEvent(parstream, parsync[0], 0));
			}
		} else {
			// attention rmsnorm
			kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(s->xb, x, w->rms_att_weight[l], dim, p->norm_eps);
		}

		// qkv matmuls for this position + RoPE encoding + update KV cache
		kernel_matmul_rope_qkv<<<(dim + kv_dim * 2) / 2, 32 * 2, 0, stream>>>(
		    PROF_TOKEN((dim + kv_dim * 2) * dim * dbits / 8), s->q, s->xb, (T*)w->wq[l], (T*)w->wk[l], (T*)w->wv[l], w->bq[l], w->bk[l], w->bv[l], dim, dim, kv_dim,
		    (KVT*)s->key_cache + loff, (KVT*)s->value_cache + loff, head_size, pos, kv_pos, kv_sink, log2(p->rope_theta), p->seq_len, p->rotary_dim);

		// only update kv cache and don't output logits
		if (l == p->n_layers - 1 && (flags & FF_UPDATE_KV_ONLY) != 0) {
			break;
		}

		size_t kvbw = p->n_kv_heads * head_size * kv_len * sizeof(KVT) + p->n_heads * kv_len * sizeof(float);

		// attention scores for all heads
		kernel_attn_score<<<dim3((kv_len + 64 * attn_par - 1) / (64 * attn_par), p->n_kv_heads), dim3(32 * attn_par, kv_mul), 0, stream>>>(
		    PROF_TOKEN(kvbw), s->att, s->q, (KVT*)s->key_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);

		// softmax the scores to get attention weights over [0..kv_len)
		kernel_attn_softmax<<<p->n_heads, softmax_size, 0, stream>>>(s->att, p->n_heads, p->seq_len, kv_len);

		// compute weighted sum of the values into xb2
		kernel_attn_mix<<<dim3(head_size / attn_par, p->n_kv_heads), dim3(32 * attn_par, kv_mul), 0, stream>>>(
		    PROF_TOKEN(kvbw), s->xb2, s->att, (KVT*)s->value_cache + loff, p->n_kv_heads, head_size, p->seq_len, kv_dim, kv_mul, kv_len);

		// final matmul to get the output of the attention
		kernel_matmul_attn<<<dim / matmul_par, 32 * matmul_par, 0, stream>>>(
		    PROF_TOKEN(dim * dim * dbits / 8), x, s->xb2, (T*)w->wo[l], dim, dim);

		if (p->arch == Mixtral) {
			// ffn rmsnorm
			kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(s->xb, x, w->rms_ffn_weight[l], dim, p->norm_eps);

			// moe gate
			assert(p->n_experts <= 32);
			float* moe_weights = s->exp + p->n_experts;
			int* moe_experts = (int*)moe_weights + p->n_experts_ac;
			kernel_moe_gate<<<1, 32 * p->n_experts, 0, stream>>>(moe_weights, moe_experts, s->xb, (T*)w->moegate[l], dim, p->n_experts, p->n_experts_ac);

			// self.w2(F.silu(self.w1(x)) * self.w3(x)) * expert weight + pre-rmsnorm residual
			kernel_matmul_moe_ffn13_silu<<<hidden_dim, dim3(32, p->n_experts_ac), 0, stream>>>(
			    PROF_TOKEN(p->n_experts_ac * 2 * hidden_dim * dim * dbits / 8), s->he, s->xb, (T**)w->moewr[l][0], (T**)w->moewr[l][2], moe_experts, dim, hidden_dim);

			kernel_matmul_moe_ffn2<<<dim, dim3(32, p->n_experts_ac), 0, stream>>>(
			    PROF_TOKEN(p->n_experts_ac * dim * hidden_dim * dbits / 8), x, s->he, (T**)w->moewr[l][1], moe_experts, moe_weights, hidden_dim, dim);
		} else if (p->arch == Phi) {
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
			kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(s->xb, x, w->rms_ffn_weight[l], dim, p->norm_eps);

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
		kernel_layernorm<<<1, layernorm_size, 0, stream>>>(x, x, s->xa, w->ln_final_weight, dim, p->norm_eps);
	} else {
		// final rmsnorm
		kernel_rmsnorm<<<1, rmsnorm_size, dim * sizeof(float), stream>>>(x, x, w->rms_final_weight, dim, p->norm_eps);
	}

	// classifier into logits
	kernel_matmul_cls<<<p->vocab_size / 32, 32 * 32, 0, stream>>>(
	    PROF_TOKEN(p->vocab_size * dim * dbits / 8), s->logits, x, (T*)w->wcls, w->bcls, dim, p->vocab_size);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors; they might fail with OOM due to lazy kernel compilation

	return s->logits;
}

extern "C" float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags) {
#define CASE(dbits_, dtype, kvbits_, kvtype)                                          \
	if (transformer->weights.dbits == dbits_ && transformer->state.kvbits == kvbits_) \
	return forward<dtype, kvtype>(transformer, token, pos, flags)

	CASE(4, uint32_t, 8, __nv_fp8_e5m2);
	CASE(4, uint32_t, 16, __half);
	CASE(8, __nv_fp8_e5m2, 8, __nv_fp8_e5m2);
	CASE(8, __nv_fp8_e5m2, 16, __half);
	CASE(16, __half, 8, __nv_fp8_e5m2);
	CASE(16, __half, 16, __half);

	assert(!"Unsupported dbits/kvbits combination for CUDA: dbits must be 4, 8 or 16, kvbits must be 8 or 16");
	return NULL;

#undef CASE
}
