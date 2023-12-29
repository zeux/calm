#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void prepare(struct Transformer* transformer) {
	struct Config* p = &transformer->config;
	struct RunState* s = &transformer->state;

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
	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrtf(ss);
	// normalize and scale
	for (int j = 0; j < size; j++) {
		o[j] = weight[j] * (ss * x[j]);
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

static void matmul(float* xout, float* x, dtype_t* w, int n, int d) {
	// W (d,n) @ x (n,) -> xout (d,)
	// by far the most amount of time is spent inside this little function
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

static void rope(float* vec, int d, int head_size, int pos, float theta) {
	for (int i = 0; i < d; i += 2) {
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

	// copy the token embedding into x
	dtype_t* content_row = w->token_embedding_table + token * dim;
	for (int i = 0; i < dim; ++i)
		x[i] = content_row[i];

	// forward all the layers
	for (unsigned long long l = 0; l < p->n_layers; l++) {

		// attention rmsnorm
		rmsnorm(s->xb, x, w->rms_att_weight[l], dim);

		// key and value point to the kv cache
		int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience

		// qkv matmuls for this position
		matmul(s->q, s->xb, w->wq[l], dim, dim);
		matmul(s->k, s->xb, w->wk[l], dim, kv_dim);
		matmul(s->v, s->xb, w->wv[l], dim, kv_dim);

		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		rope(s->q, dim, head_size, pos, p->rope_theta);
		rope(s->k, kv_dim, head_size, pos, p->rope_theta);

		// update kv cache
		for (int i = 0; i < kv_dim; i++) {
			s->key_cache[loff + pos * kv_dim + i] = s->k[i];
			s->value_cache[loff + pos * kv_dim + i] = s->v[i];
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
			for (int t = 0; t <= pos; t++) {
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

			// softmax the scores to get attention weights, from 0..pos inclusively
			softmax(att, pos + 1);

			// weighted sum of the values, store back into xb
			float* xb = s->xb + h * head_size;
			for (int i = 0; i < head_size; i++) {
				xb[i] = 0.0f;
			}
			for (int t = 0; t <= pos; t++) {
				// get the value vector for this head and at this timestep
				kvtype_t* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
				// get the attention weight for this timestep
				float a = att[t];
				// accumulate the weighted value into xb
				for (int i = 0; i < head_size; i++) {
					xb[i] += a * v[i];
				}
			}
		}

		// final matmul to get the output of the attention
		matmul(s->xb2, s->xb, w->wo[l], dim, dim);

		// residual connection back into x
		for (int i = 0; i < dim; i++) {
			x[i] += s->xb2[i];
		}

		// ffn rmsnorm
		rmsnorm(s->xb, x, w->rms_ffn_weight[l], dim);

		// Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
		// first calculate self.w1(x) and self.w3(x)
		matmul(s->hb, s->xb, w->w1[l], dim, hidden_dim);
		matmul(s->hb2, s->xb, w->w3[l], dim, hidden_dim);

		// SwiGLU non-linearity
		for (int i = 0; i < hidden_dim; i++) {
			float val = s->hb[i];
			// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
			val *= (1.0f / (1.0f + expf(-val)));
			// elementwise multiply with w3(x)
			val *= s->hb2[i];
			s->hb[i] = val;
		}

		// final matmul to get the output of the ffn
		matmul(s->xb, s->hb, w->w2[l], hidden_dim, dim);

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
	rmsnorm(x, x, w->rms_final_weight, dim);

	// classifier into logits
	matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
	return s->logits;
}
