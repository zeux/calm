#include "sampler.h"

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

static int sample_argmax(float* probabilities, int n) {
	// return the index that has the highest probability
	int max_i = 0;
	float max_p = probabilities[0];
	for (int i = 1; i < n; i++) {
		if (probabilities[i] > max_p) {
			max_i = i;
			max_p = probabilities[i];
		}
	}
	return max_i;
}

static int sample_cutoff(float* probabilities, int n, float cutoff, float coin) {
	int n0 = 0;
	float cumulative_prob = 0.0f;
	for (int i = 0; i < n; i++) {
		if (probabilities[i] >= cutoff) {
			cumulative_prob += probabilities[i];
			n0 = i; // for fallback due to rounding errors
		} else {
			probabilities[i] = 0.0f;
		}
	}

	// sample from the truncated list
	float r = coin * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i < n; i++) {
		cdf += probabilities[i];
		if (r < cdf) {
			return i;
		}
	}
	return n0; // in case of rounding errors
}

int sample(struct Sampler* sampler, float* logits) {
	if (sampler->temperature == 0.0f || sampler->minp >= 1.0f) {
		// greedy argmax sampling: take the token with the highest probability
		return sample_argmax(logits, sampler->vocab_size);
	} else {
		// apply softmax to the logits to get the probabilities for next token
		float maxp = sample_softmax(logits, sampler->vocab_size, 1.0f / sampler->temperature);
		// flip a (float) coin (this is our source of entropy for sampling)
		float coin = random_f32(&sampler->rng_state);
		// min-p (cutoff) sampling, clamping the least likely tokens to zero
		return sample_cutoff(logits, sampler->vocab_size, sampler->minp * maxp, coin);
	}
}
