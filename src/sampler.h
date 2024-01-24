#pragma once

struct Sampler {
	int vocab_size;
	unsigned long long rng_state;

	float temperature;
	float minp;
};

float sample_prob(int idx, float* logits, int size);

int sample(struct Sampler* sampler, float* logits);
