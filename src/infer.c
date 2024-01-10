#include "model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif

// we only support CPU inference when the compiler supports _Float16 type natively
#if defined(__FLT16_MANT_DIG__)
typedef _Float16 half;
#else
typedef short half;
#endif

static half fp82half(unsigned char v) {
	union {
		unsigned short u;
		half f;
	} u;
	u.u = v << 8;
	return u.f;
}

typedef float (*dotprod_t)(void* w, int n, int i, float* x);

static float dotprod_fp16(void* w, int n, int i, float* x) {
	half* r = (half*)w + i * n;
#if defined(__AVX2__) && defined(__F16C__)
	assert(n % 16 == 0);
	__m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
	for (int j = 0; j < n; j += 16) {
		__m256i rw = _mm256_loadu_si256((__m256i*)&r[j]);
		__m128i rlo = _mm256_castsi256_si128(rw);
		__m128i rhi = _mm256_extractf128_si256(rw, 1);
		__m256 x0 = _mm256_loadu_ps(&x[j]);
		__m256 x1 = _mm256_loadu_ps(&x[j + 8]);
		acc0 = _mm256_add_ps(_mm256_mul_ps(x0, _mm256_cvtph_ps(rlo)), acc0);
		acc1 = _mm256_add_ps(_mm256_mul_ps(x1, _mm256_cvtph_ps(rhi)), acc1);
	}
	__m256 acc8 = _mm256_add_ps(acc0, acc1);
	__m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
	__m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
	return _mm_cvtss_f32(accf);
#else
	float val = 0.0f;
#pragma omp simd reduction(+ : val) simdlen(32)
	for (int j = 0; j < n; j++) {
		val += r[j] * x[j];
	}
	return val;
#endif
}

static float dotprod_fp8(void* w, int n, int i, float* x) {
	char* r = (char*)w + i * n;
#if defined(__AVX2__) && defined(__F16C__)
	assert(n % 16 == 0);
	__m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
	for (int j = 0; j < n; j += 16) {
		__m128i rw = _mm_loadu_si128((__m128i*)&r[j]);
		__m128i rlo = _mm_unpacklo_epi8(_mm_setzero_si128(), rw);
		__m128i rhi = _mm_unpackhi_epi8(_mm_setzero_si128(), rw);
		__m256 x0 = _mm256_loadu_ps(&x[j]);
		__m256 x1 = _mm256_loadu_ps(&x[j + 8]);
		acc0 = _mm256_add_ps(_mm256_mul_ps(x0, _mm256_cvtph_ps(rlo)), acc0);
		acc1 = _mm256_add_ps(_mm256_mul_ps(x1, _mm256_cvtph_ps(rhi)), acc1);
	}
	__m256 acc8 = _mm256_add_ps(acc0, acc1);
	__m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
	__m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
	return _mm_cvtss_f32(accf);
#else
	float val = 0.0f;
#pragma omp simd reduction(+ : val) simdlen(32)
	for (int j = 0; j < n; j++) {
		val += fp82half(r[j]) * x[j];
	}
	return val;
#endif
}

static dotprod_t dotprod;

void prepare(struct Transformer* transformer) {
	struct Config* p = &transformer->config;
	struct RunState* s = &transformer->state;

	dotprod = transformer->weights.dsize == 1 ? dotprod_fp8 : dotprod_fp16;

	// we calloc instead of malloc to keep valgrind happy
	int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
	s->x = calloc(p->dim, sizeof(float));
	s->xb = calloc(p->dim, sizeof(float));
	s->xb2 = calloc(p->dim, sizeof(float));
	s->hb = calloc(p->hidden_dim, sizeof(float));
	s->hb2 = calloc(p->hidden_dim, sizeof(float));
	s->q = calloc(p->dim, sizeof(float));
	s->k = calloc(kv_dim, sizeof(float));
	s->v = calloc(kv_dim, sizeof(float));
	s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
	s->logits = calloc(p->vocab_size, sizeof(float));
	s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kvtype_t));
	s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(kvtype_t));
	// ensure all mallocs went fine
	if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
		fprintf(stderr, "malloc failed!\n");
		abort();
	}

#if defined(_OPENMP) && defined(__linux__)
	// avoid SMT overhead by default
	if (getenv("OMP_NUM_THREADS") == NULL) {
		omp_set_num_threads(omp_get_num_procs() / 2);
	}
#endif

#if !defined(__FLT16_MANT_DIG__)
	fprintf(stderr, "FATAL: _Float16 compiler support is required for CPU backend\n");
	abort();
#endif
}

static void rmsnorm(float* o, float* x, float* weight, int size) {
	// calculate sum of squares
	float ss = 0.0f;
	for (int j = 0; j < size; j++) {
		ss += x[j] * x[j];
	}
	// normalize and scale
	float scale = 1.0f / sqrtf(ss / size + 1e-5f);
	for (int j = 0; j < size; j++) {
		o[j] = weight[j] * (scale * x[j]);
	}
}

static void layernorm(float* o, float* x, float* weight, float* bias, int size) {
	// calculate sum
	float ss = 0.0f;
	for (int j = 0; j < size; j++) {
		ss += x[j];
	}

	float mean = ss / size;

	// calculate sum of squared deltas
	ss = 0.0f;
	for (int j = 0; j < size; j++) {
		ss += (x[j] - mean) * (x[j] - mean);
	}

	float var = ss / size;

	// normalize and scale
	float scale = 1.0f / sqrtf(var + 1e-5f);
	for (int j = 0; j < size; j++) {
		o[j] = (x[j] - mean) * scale * weight[j] + bias[j];
	}
}

static void softmax(float* x, int size) {
	// find max value (for numerical stability)
	float max_val = x[0];
	for (int i = 1; i < size; i++) {
		if (x[i] > max_val) {
			max_val = x[i];
		}
	}
	// exp and sum
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}
	// normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
}

static void matmul(float* xout, float* x, void* w, float* b, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = dotprod(w, n, i, x);
		if (b) {
			val += b[i];
		}
		xout[i] = val;
	}
}

static void rope(float* vec, int d, int head_size, int pos, float theta, int rotary_dim) {
	for (int i = 0; i < d; i += 2) {
		int j_head = i % head_size;
		float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
		float val = pos * freq;
		float fcr = cosf(val);
		float fci = sinf(val);

		float v0 = vec[i];
		float v1 = vec[i + 1];
		vec[i] = v0 * fcr - v1 * fci;
		vec[i + 1] = v0 * fci + v1 * fcr;
	}
}

float* forward(struct Transformer* transformer, int token, int pos, unsigned flags) {

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

	// copy the token embedding into x
	char* content_row = (char*)w->token_embedding_table + token * dim * w->dsize;
	for (int i = 0; i < dim; ++i) {
		x[i] = w->dsize == 1 ? fp82half(content_row[i]) : ((half*)content_row)[i];
	}

	// forward all the layers
	for (unsigned long long l = 0; l < p->n_layers; l++) {

		if (p->arch == Phi) {
			// input layernorm
			layernorm(s->xb, x, w->ln_weight[l], w->ln_bias[l], dim);
		} else {
			// attention rmsnorm
			rmsnorm(s->xb, x, w->rms_att_weight[l], dim);
		}

		// key and value point to the kv cache
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		// qkv matmuls for this position
		matmul(s->q, s->xb, w->wq[l], w->bq[l], dim, dim);
		matmul(s->k, s->xb, w->wk[l], w->bk[l], dim, kv_dim);
		matmul(s->v, s->xb, w->wv[l], w->wv[l], dim, kv_dim);

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		rope(s->q, dim, head_size, pos, p->rope_theta, p->rotary_dim);
		rope(s->k, kv_dim, head_size, pos, p->rope_theta, p->rotary_dim);

		// update kv cache
		for (int i = 0; i < kv_dim; i++) {
			s->key_cache[loff + kv_pos * kv_dim + i] = s->k[i];
			s->value_cache[loff + kv_pos * kv_dim + i] = s->v[i];
		}

		// rotate sink tokens forward to keep pace with non-sink tokens
		for (int r = 0; r < kv_sink; r++) {
			for (int i = 0; i < kv_dim; i++) {
				s->k[i] = s->key_cache[loff + r * kv_dim + i];
			}

			rope(s->k, kv_dim, head_size, 1, p->rope_theta, p->rotary_dim);

			for (int i = 0; i < kv_dim; i++) {
				s->key_cache[loff + r * kv_dim + i] = s->k[i];
			}
		}

		// multihead attention. iterate over all heads
		int h;
#pragma omp parallel for private(h)
		for (h = 0; h < p->n_heads; h++) {
			// get the query vector for this head
			float* q = s->q + h * head_size;
			// attention scores for this head
			float* att = s->att + h * p->seq_len;
			// iterate over all timesteps, including the current one
			for (int t = 0; t < kv_len; t++) {
				// get the key vector for this head and at this timestep
				kvtype_t* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// calculate the attention score as the dot product of q and k
				float score = 0.0f;
				for (int i = 0; i < head_size; i++) {
					score += q[i] * k[i];
				}
				score /= sqrtf(head_size);
				// save the score to the attention buffer
				att[t] = score;
			}

			// softmax the scores to get attention weights over [0..kv_len)
			softmax(att, kv_len);

			// weighted sum of the values, store back into xb2
			float* xb2 = s->xb2 + h * head_size;
			for (int i = 0; i < head_size; i++) {
				xb2[i] = 0.0f;
			}
			for (int t = 0; t < kv_len; t++) {
				// get the value vector for this head and at this timestep
				kvtype_t* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// get the attention weight for this timestep
				float a = att[t];
				// accumulate the weighted value into xb
				for (int i = 0; i < head_size; i++) {
					xb2[i] += a * v[i];
				}
			}
		}

		// final matmul to get the output of the attention
		// TODO: we're using hb as a temporary storage, hacky
		matmul(s->hb, s->xb2, w->wo[l], w->bo[l], dim, dim);

		// residual connection back into x
		for (int i = 0; i < dim; i++) {
			x[i] += s->hb[i];
		}

		if (p->arch == Phi) {
			matmul(s->hb, s->xb, w->w1[l], w->b1[l], dim, hidden_dim);

			// GELU non-linearity
			for (int i = 0; i < hidden_dim; i++) {
				float val = s->hb[i];
				// GELU (0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))))
				val = 0.5f * val * (1.0f + tanhf(0.797885f * (val + 0.044715f * val * val * val)));
				s->hb[i] = val;
			}
		} else {
			// ffn rmsnorm
			rmsnorm(s->xb, x, w->rms_ffn_weight[l], dim);

			// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
			// first calculate self.w1(x) and self.w3(x)
			matmul(s->hb, s->xb, w->w1[l], NULL, dim, hidden_dim);
			matmul(s->hb2, s->xb, w->w3[l], NULL, dim, hidden_dim);

			// SwiGLU non-linearity
			for (int i = 0; i < hidden_dim; i++) {
				float val = s->hb[i];
				// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
				val *= (1.0f / (1.0f + expf(-val)));
				// elementwise multiply with w3(x)
				val *= s->hb2[i];
				s->hb[i] = val;
			}
		}

		// final matmul to get the output of the ffn
		matmul(s->xb, s->hb, w->w2[l], w->b2[l], hidden_dim, dim);

		// residual connection
		for (int i = 0; i < dim; i++) {
			x[i] += s->xb[i];
		}
	}

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	if (p->arch == Phi) {
		// final layernorm
		layernorm(x, x, w->ln_final_weight, w->ln_final_bias, dim);
	} else {
		// final rmsnorm
		rmsnorm(x, x, w->rms_final_weight, dim);
	}

	// classifier into logits
	matmul(s->logits, x, w->wcls, w->bcls, p->dim, p->vocab_size);

	return s->logits;
}
