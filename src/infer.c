#include "model.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
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

// we only support fp16 kv cache by default; this can be changed to float with a recompile
typedef half kvtype_t;

inline half fp82half(unsigned char v) {
	union {
		unsigned short u;
		half f;
	} u;
	u.u = v << 8;
	return u.f;
}

inline float gf4_ff(uint32_t v, int k) {
	float s = fp82half(v & 0xff) / -4.f; // we expect compiler to reuse this across multiple calls
	return ((int)((v >> (8 + k * 3)) & 7) - 4) * s;
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

static float dotprod_gf4(void* w, int n, int i, float* x) {
	uint32_t* r = (uint32_t*)w + i * n / 8;
#if defined(__AVX2__) && defined(__F16C__)
	assert(n % 32 == 0);
	__m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
	for (int j = 0; j < n; j += 32) {
		__m128i wg = _mm_loadu_si128((__m128i*)&r[j / 8]);
		const __m128i wgfm = _mm_setr_epi8(-1, 0, -1, 4, -1, 8, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1);
		__m128 wgf = _mm_cvtph_ps(_mm_shuffle_epi8(wg, wgfm)); // note: scale 1/-4.f is baked into wgtab below
		__m256 x0 = _mm256_loadu_ps(&x[j]);
		__m256 x1 = _mm256_loadu_ps(&x[j + 8]);
		__m256 x2 = _mm256_loadu_ps(&x[j + 16]);
		__m256 x3 = _mm256_loadu_ps(&x[j + 24]);
		__m256i wgp = _mm256_broadcastsi128_si256(wg);
		__m256 wgfp = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(wgf)));
		const __m256i wgbits = _mm256_setr_epi32(8, 11, 14, 17, 20, 23, 26, 29);
		const __m256 wgtab = _mm256_setr_ps(-4 / -4.f, -3 / -4.f, -2 / -4.f, -1 / -4.f, 0 / -4.f, 1 / -4.f, 2 / -4.f, 3 / -4.f);
		__m256 w0 = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0x00), wgbits));
		__m256 w1 = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0x55), wgbits));
		__m256 w2 = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0xaa), wgbits));
		__m256 w3 = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0xff), wgbits));
		acc0 = _mm256_add_ps(_mm256_mul_ps(w0, _mm256_mul_ps(x0, _mm256_shuffle_ps(wgfp, wgfp, 0x00))), acc0);
		acc1 = _mm256_add_ps(_mm256_mul_ps(w1, _mm256_mul_ps(x1, _mm256_shuffle_ps(wgfp, wgfp, 0x55))), acc1);
		acc0 = _mm256_add_ps(_mm256_mul_ps(w2, _mm256_mul_ps(x2, _mm256_shuffle_ps(wgfp, wgfp, 0xaa))), acc0);
		acc1 = _mm256_add_ps(_mm256_mul_ps(w3, _mm256_mul_ps(x3, _mm256_shuffle_ps(wgfp, wgfp, 0xff))), acc1);
	}
	__m256 acc8 = _mm256_add_ps(acc0, acc1);
	__m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
	__m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
	return _mm_cvtss_f32(accf);
#else
	float val = 0.0f;
	for (int j = 0; j < n; j += 8) {
		uint32_t wg = r[j / 8];
		for (int k = 0; k < 8; ++k) {
			val += gf4_ff(wg, k) * x[j + k];
		}
	}
	return val;
#endif
}

void prepare(struct Transformer* transformer) {
	struct Config* p = &transformer->config;
	struct RunState* s = &transformer->state;

	int q_dim = p->head_dim * p->n_heads;
	int kv_dim = p->head_dim * p->n_kv_heads;

	// we calloc instead of malloc to keep valgrind happy
	s->x = calloc(p->dim, sizeof(float));
	s->xb = calloc(p->dim, sizeof(float));
	s->xb2 = calloc(p->dim, sizeof(float));
	s->hb = calloc(p->hidden_dim, sizeof(float));
	s->hb2 = calloc(p->hidden_dim, sizeof(float));
	s->q = calloc(q_dim, sizeof(float));
	s->k = calloc(kv_dim, sizeof(float));
	s->v = calloc(kv_dim, sizeof(float));
	s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
	s->exp = calloc(p->n_experts + (p->n_experts_ac ? p->n_experts_ac : 1) * 2, sizeof(float));
	s->logits = calloc(p->vocab_size, sizeof(float));
	assert(s->kvbits == sizeof(kvtype_t) * 8);
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
	assert(!"_Float16 compiler support is required for CPU backend\n");
#endif
}

static void rmsnorm(float* o, float* x, float* weight, int size, float eps, bool ln) {
	// calculate mean
	float mean = 0.0f;

	if (ln) {
		for (int j = 0; j < size; j++) {
			mean += x[j];
		}
		mean /= size;
	}

	// calculate sum of squared deltas
	float ss = 0.0f;
	for (int j = 0; j < size; j++) {
		ss += (x[j] - mean) * (x[j] - mean);
	}

	float var = ss / size;

	// normalize and scale
	float scale = 1.0f / sqrtf(var + eps);
	for (int j = 0; j < size; j++) {
		o[j] = (x[j] - mean) * scale * weight[j];
	}
}

static void matmul(float* xout, float* x, void* w, float* b, int n, int d, dotprod_t dotprod) {
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

static void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
	for (int i = 0; i < d; i += 2) {
		int j_head = i % head_dim;
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

static void attn(float* xout, float* atth, float* qh, kvtype_t* kh, kvtype_t* vh, int head_dim, int kv_dim, int kv_len) {
	float score_max = -FLT_MAX;

	// calculate attention scores as dot products of q and k; also track score max for this head
	for (int t = 0; t < kv_len; ++t) {
		float score = 0.0f;
		for (int j = 0; j < head_dim; ++j) {
			score += qh[j] * kh[t * kv_dim + j];
		}
		score /= sqrtf(head_dim);
		score_max = (score_max < score) ? score : score_max;
		atth[t] = score;
	}

	// softmax the scores to get attention weights over [0..kv_len)
	float score_sum = 0.f;
	for (int t = 0; t < kv_len; ++t) {
		atth[t] = expf(atth[t] - score_max);
		score_sum += atth[t];
	}

	// mix values with attention weights
	for (int j = 0; j < head_dim; ++j) {
		float res = 0.f;
		for (int t = 0; t < kv_len; ++t) {
			res += (atth[t] / score_sum) * vh[t * kv_dim + j];
		}
		xout[j] = res;
	}
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

static void moe_gate(float* moe_weights, int* moe_experts, float* x, int d, int active) {
	// softmax across experts
	float max_val = -FLT_MAX;
	for (int j = 0; j < d; ++j) {
		max_val = (max_val < x[j]) ? x[j] : max_val;
	}

	// top k
	uint64_t mask = 0;
	float wsum = 0.0f;

	for (int k = 0; k < active; ++k) {
		int best = -1;
		for (int j = 0; j < d; ++j) {
			if ((mask & (1ull << j)) == 0 && (best == -1 || x[j] > x[best])) {
				best = j;
			}
		}

		moe_experts[k] = best;
		wsum += expf(x[moe_experts[k]] - max_val);
		mask |= 1ull << best;
	}

	// top k weights, normalized
	for (int k = 0; k < active; ++k) {
		moe_weights[k] = expf(x[moe_experts[k]] - max_val) / wsum;
	}
}

float* forward(struct Transformer* transformer, int token, int pos, unsigned flags) {
	if (transformer->weights.dbits != 4 && transformer->weights.dbits != 8 && transformer->weights.dbits != 16) {
		assert(!"Unsupported dbits: must be 8 or 16 for CPU");
	}

	dotprod_t dotprod = transformer->weights.dbits == 4 ? dotprod_gf4 : (transformer->weights.dbits == 8 ? dotprod_fp8 : dotprod_fp16);

	// a few convenience variables
	struct Config* p = &transformer->config;
	struct Weights* w = &transformer->weights;
	struct RunState* s = &transformer->state;
	float* x = s->x;
	int dim = p->dim;
	int hidden_dim = p->hidden_dim;
	int q_dim = p->head_dim * p->n_heads;
	int kv_dim = p->head_dim * p->n_kv_heads;
	int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery

	// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
	int kv_sink = pos >= p->seq_len ? KV_SINKS : 0;
	int kv_pos = kv_sink + (pos - kv_sink) % (p->seq_len - kv_sink);
	int kv_len = pos >= p->seq_len ? p->seq_len : pos + 1;

	// copy the token embedding into x
	char* content_row = (char*)w->token_embedding_table + token * dim * (size_t)w->dbits / 8;
	if (w->dbits == 4) {
		for (int i = 0; i < dim; i += 8) {
			uint32_t wg = ((uint32_t*)content_row)[i / 8];
			for (int k = 0; k < 8; ++k) {
				x[i + k] = gf4_ff(wg, k);
			}
		}
	} else {
		for (int i = 0; i < dim; ++i) {
			x[i] = w->dbits == 8 ? fp82half(content_row[i]) : ((half*)content_row)[i];
		}
	}

	for (int i = 0; i < dim; ++i) {
		x[i] *= p->embed_scale;
	}

	// forward all the layers
	for (int l = 0; l < p->n_layers; l++) {

		// attention rmsnorm
		rmsnorm(s->xb, x, w->rms_att_weight[l], dim, p->norm_eps, p->norm_mean);

		// key and value point to the kv cache
		size_t loff = (size_t)l * p->seq_len * kv_dim; // kv cache layer offset for convenience
		kvtype_t* kb = (kvtype_t*)s->key_cache + loff;
		kvtype_t* vb = (kvtype_t*)s->value_cache + loff;

		// qkv matmuls for this position
		matmul(s->q, s->xb, w->wq[l], w->bqkv[l], dim, q_dim, dotprod);
		matmul(s->k, s->xb, w->wk[l], w->bqkv[l] ? w->bqkv[l] + q_dim : NULL, dim, kv_dim, dotprod);
		matmul(s->v, s->xb, w->wv[l], w->bqkv[l] ? w->bqkv[l] + q_dim + kv_dim : NULL, dim, kv_dim, dotprod);

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		rope(s->q, q_dim, p->head_dim, pos, p->rope_theta, p->rotary_dim);
		rope(s->k, kv_dim, p->head_dim, pos, p->rope_theta, p->rotary_dim);

		// update kv cache
		for (int i = 0; i < kv_dim; i++) {
			kb[kv_pos * kv_dim + i] = s->k[i];
			vb[kv_pos * kv_dim + i] = s->v[i];
		}

		// rotate sink tokens forward to keep pace with non-sink tokens
		for (int r = 0; r < kv_sink; r++) {
			for (int i = 0; i < kv_dim; i++) {
				s->k[i] = kb[r * kv_dim + i];
			}

			rope(s->k, kv_dim, p->head_dim, 1, p->rope_theta, p->rotary_dim);

			for (int i = 0; i < kv_dim; i++) {
				kb[r * kv_dim + i] = s->k[i];
			}
		}

		// multihead attention. iterate over all heads
		int h;
#pragma omp parallel for private(h)
		for (h = 0; h < p->n_heads; h++) {
			float* qh = s->q + h * p->head_dim;
			float* atth = s->att + h * p->seq_len;
			kvtype_t* kh = kb + (h / kv_mul) * p->head_dim;
			kvtype_t* vh = vb + (h / kv_mul) * p->head_dim;

			attn(s->xb2 + h * p->head_dim, atth, qh, kh, vh, p->head_dim, kv_dim, kv_len);
		}

		// final matmul to get the output of the attention
		// TODO: we're using hb as a temporary storage, hacky
		matmul(s->hb, s->xb2, w->wo[l], NULL, q_dim, dim, dotprod);

		// residual connection back into x
		for (int i = 0; i < dim; i++) {
			x[i] += s->hb[i];
		}

		// ffn rmsnorm
		rmsnorm(s->xb, x, w->rms_ffn_weight[l], dim, p->norm_eps, p->norm_mean);

		float* moe_weights = s->exp + p->n_experts;
		int* moe_experts = (int*)moe_weights + p->n_experts_ac;

		if (p->n_experts) {
			// moe gate
			matmul(s->exp, s->xb, w->moegate[l], NULL, dim, p->n_experts, dotprod);
			moe_gate(moe_weights, moe_experts, s->exp, p->n_experts, p->n_experts_ac);
		} else {
			moe_weights[0] = 1.0f;
			moe_experts[0] = 0;
		}

		// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
		for (int e = 0; e < (p->n_experts_ac ? p->n_experts_ac : 1); ++e) {
			size_t esize = dim * hidden_dim * (size_t)w->dbits / 8;
			matmul(s->hb, s->xb, (char*)w->w1[l] + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);
			matmul(s->hb2, s->xb, (char*)w->w3[l] + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);

			if (p->act_gelu) {
				// GEGLU non-linearity
				for (int i = 0; i < hidden_dim; i++) {
					s->hb[i] = gelu(s->hb[i]) * s->hb2[i];
				}
			} else {
				// SwiGLU non-linearity
				for (int i = 0; i < hidden_dim; i++) {
					s->hb[i] = silu(s->hb[i]) * s->hb2[i];
				}
			}

			matmul(s->xb2, s->hb, (char*)w->w2[l] + moe_experts[e] * esize, NULL, hidden_dim, dim, dotprod);

			for (int i = 0; i < dim; i++) {
				x[i] += s->xb2[i] * moe_weights[e];
			}
		}

		// final matmul to get the output of the ffn
		matmul(s->xb, s->hb, w->w2[l], NULL, hidden_dim, dim, dotprod);

		// residual connection
		for (int i = 0; i < dim; i++) {
			x[i] += s->xb[i];
		}
	}

	if (flags & FF_UPDATE_KV_ONLY) {
		// only update kv cache and don't output logits
		return NULL;
	}

	// final rmsnorm
	rmsnorm(x, x, w->rms_final_weight, dim, p->norm_eps, p->norm_mean);

	// classifier into logits
	matmul(s->logits, x, w->wcls, NULL, p->dim, p->vocab_size, dotprod);

	return s->logits;
}
