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

void sample_softmax(float* x, int size) {
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

static int sample_minp(float* probabilities, int n, float minp, struct ProbIndex* probindex, float coin) {
	float maxprob = 0.f;
	for (int i = 0; i < n; ++i) {
		maxprob = maxprob < probabilities[i] ? probabilities[i] : maxprob;
	}

	float cutoff = maxprob * minp;

	int n0 = 0;
	float cumulative_prob = 0.0f;
	for (int i = 0; i < n; i++) {
		if (probabilities[i] >= cutoff) {
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			cumulative_prob += probabilities[i];
			n0++;
		}
	}

	// sample from the truncated list
	float r = coin * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i < n0; i++) {
		cdf += probindex[i].prob;
		if (r < cdf) {
			return probindex[i].index;
		}
	}
	return probindex[n0 - 1].index; // in case of rounding errors
}

void sampler_init(struct Sampler* sampler, int vocab_size, float temperature, float minp, unsigned long long rng_seed) {
	sampler->vocab_size = vocab_size;
	sampler->temperature = temperature;
	sampler->minp = minp;
	sampler->rng_state = rng_seed;
	// buffer only used with nucleus sampling; may not need but it's ~small
	sampler->probindex = malloc(sampler->vocab_size * sizeof(struct ProbIndex));
}

void sampler_free(struct Sampler* sampler) {
	free(sampler->probindex);
}

int sample(struct Sampler* sampler, float* logits) {
	if (sampler->temperature == 0.0f || sampler->minp >= 1.0f) {
		// greedy argmax sampling: take the token with the highest probability
		return sample_argmax(logits, sampler->vocab_size);
	} else {
		// apply the temperature to the logits
		for (int q = 0; q < sampler->vocab_size; q++) {
			logits[q] /= sampler->temperature;
		}
		// apply softmax to the logits to get the probabilities for next token
		sample_softmax(logits, sampler->vocab_size);
		// flip a (float) coin (this is our source of entropy for sampling)
		float coin = random_f32(&sampler->rng_state);
		// min-p (cutoff) sampling, clamping the least likely tokens to zero
		return sample_minp(logits, sampler->vocab_size, sampler->minp, sampler->probindex, coin);
	}
}
