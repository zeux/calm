#pragma once

struct Sampler {
	int vocab_size;
	unsigned long long rng_state;

	float temperature;
	float minp;
};

float sample_softmax(float* x, int size, float scale);

int sample(struct Sampler* sampler, float* logits);
