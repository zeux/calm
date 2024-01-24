#include "sampler.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

static unsigned int random_u32(unsigned long long* state) {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long* state) { // random float32 in [0,1)
	return (random_u32(state) >> 8) / 16777216.0f;
}

float sample_softmax(float* x, int size, float scale) {
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
		x[i] = expf((x[i] - max_val) * scale);
		sum += x[i];
	}
	// normalize
	for (int i = 0; i < size; i++) {
		x[i] /= sum;
	}
	// return max value (post-softmax)
	return 1.0f / sum;
}

static int sample_argmax(float* logits, int n) {
	int max_i = -1;
	float max_p = -FLT_MAX;
	for (int i = 0; i < n; i++) {
		max_i = logits[i] > max_p ? i : max_i;
		max_p = logits[i] > max_p ? logits[i] : max_p;
	}
	return max_i;
}

static int sample_minp(float* logits, int n, float minp, float temperature, float coin) {
	// find max logit; we will use this to derive minp cutoff (in log space), since minp is scale-invariant (wrt softmax)
	float max_logit = -FLT_MAX;
	for (int i = 0; i < n; i++) {
		max_logit = logits[i] > max_logit ? logits[i] : max_logit;
	}

	// exp(logit / temp) <= exp(max_logit / temp) * minp -> logit <= max_logit + log(minp) * temp
	float logit_cutoff = max_logit + logf(minp) * temperature;

	// convert from logits to probabilities in-place while simultaneously doing (unscaled) softmax; we'll rescale later
	float* probs = logits;
	int fallback = 0;
	float cumulative_prob = 0.0f;
	for (int i = 0; i < n; i++) {
		if (logits[i] >= logit_cutoff) {
			probs[i] = expf((logits[i] - max_logit) / temperature);
			cumulative_prob += probs[i];
			fallback = i; // for fallback due to rounding errors
		} else {
			probs[i] = 0.0f;
		}
	}

	// sample from the truncated list
	float r = coin * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i < n; i++) {
		cdf += probs[i];
		if (r < cdf) {
			return i;
		}
	}
	return fallback; // in case of rounding errors
}

int sample(struct Sampler* sampler, float* logits) {
	if (sampler->temperature == 0.0f || sampler->minp >= 1.0f) {
		// greedy argmax sampling: take the token with the highest probability
		return sample_argmax(logits, sampler->vocab_size);
	} else {
		float coin = random_f32(&sampler->rng_state);
		// min-p (cutoff) sampling, clamping the least likely tokens to zero
		return sample_minp(logits, sampler->vocab_size, sampler->minp, sampler->temperature, coin);
	}
}
