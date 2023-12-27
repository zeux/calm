#pragma once

struct ProbIndex {
	float prob;
	int index;
};

struct Sampler {
	int vocab_size;
	float temperature;
	float topp;
	unsigned long long rng_state;

	struct ProbIndex* probindex; // buffer used in top-p sampling
};

void sampler_init(struct Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void sampler_free(struct Sampler* sampler);

int sample(struct Sampler* sampler, float* logits);
