// Inference for Llama-2 Transformer model in pure C
// Based on llama2.c by Andrej Karpathy

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "profiler.h"
#include "sampler.h"
#include "tensors.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// Transformer model

void prepare(struct Transformer* transformer);
float* forward(struct Transformer* transformer, int token, int pos, unsigned flags);

void prepare_cuda(struct Transformer* transformer);
float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags);

void build_transformer(struct Config* config, struct Weights* weights, struct Tensors* tensors) {
	// create config
	config->dim = atoi(tensors_metadata(tensors, "dim"));
	config->hidden_dim = atoi(tensors_metadata(tensors, "hidden_dim"));
	config->n_layers = atoi(tensors_metadata(tensors, "n_layers"));
	config->n_heads = atoi(tensors_metadata(tensors, "n_heads"));
	config->n_kv_heads = atoi(tensors_metadata(tensors, "n_kv_heads"));
	config->vocab_size = atoi(tensors_metadata(tensors, "vocab_size"));
	config->seq_len = 4096;

	const char* rope_theta = tensors_metadata_find(tensors, "rope_theta");
	config->rope_theta = rope_theta ? atof(rope_theta) : 10000.f;

	int head_size = config->dim / config->n_heads;

	// get tensor data
	enum DType dtype = dt_f16;

	weights->token_embedding_table = (dtype_t*)tensors_get(tensors, "model.embed.weight", 0, dtype, (int[]){config->vocab_size, config->dim, 0, 0});

	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (float*)tensors_get(tensors, "model.layers.%d.attn.norm.weight", l, dt_f32, (int[]){config->dim, 0, 0, 0});
		weights->wq[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.attn.wq.weight", l, dtype, (int[]){config->dim, config->n_heads * head_size, 0, 0});
		weights->wk[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.attn.wk.weight", l, dtype, (int[]){config->n_kv_heads * head_size, config->dim, 0, 0});
		weights->wv[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.attn.wv.weight", l, dtype, (int[]){config->n_kv_heads * head_size, config->dim, 0, 0});
		weights->wo[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.attn.wo.weight", l, dtype, (int[]){config->n_heads * head_size, config->dim, 0, 0});

		weights->rms_ffn_weight[l] = (float*)tensors_get(tensors, "model.layers.%d.mlp.norm.weight", l, dt_f32, (int[]){config->dim, 0, 0, 0});
		weights->w1[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.w1.weight", l, dtype, (int[]){config->hidden_dim, config->dim, 0, 0});
		weights->w2[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.w2.weight", l, dtype, (int[]){config->dim, config->hidden_dim, 0, 0});
		weights->w3[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.w3.weight", l, dtype, (int[]){config->hidden_dim, config->dim, 0, 0});
	}

	weights->rms_final_weight = (float*)tensors_get(tensors, "model.norm.weight", 0, dt_f32, (int[]){config->dim, 0, 0, 0});
	weights->wcls = (dtype_t*)tensors_get(tensors, "model.output.weight", 0, dtype, (int[]){config->vocab_size, config->dim, 0, 0});
}

void build_tokenizer(struct Tokenizer* t, struct Tensors* tensors, int vocab_size) {
	struct Tensor* tensor = tensors_find(tensors, "tokenizer.tokens", 0);

	char* tokens = (char*)tensors_get(tensors, "tokenizer.tokens", 0, dt_u8, (int[]){tensor->shape[0], 0, 0, 0});
	float* scores = (float*)tensors_get(tensors, "tokenizer.scores", 0, dt_f32, (int[]){vocab_size, 0, 0, 0});

	int bos_id = atoi(tensors_metadata(tensors, "bos_token_id"));
	int eos_id = atoi(tensors_metadata(tensors, "eos_token_id"));

	tokenizer_init(t, tokens, scores, bos_id, eos_id, vocab_size);
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
	// return time in milliseconds, for benchmarking the model speed
	struct timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

size_t model_bandwidth(struct Config* config) {
	int head_size = config->dim / config->n_heads;

	size_t res = 0;

	// token embedding table (vocab_size, dim)
	res += sizeof(dtype_t) * config->vocab_size * config->dim;
	// weights for rmsnorms (dim) x 2 x layers
	res += sizeof(dtype_t) * config->dim * 2 * config->n_layers;
	// weights for matmuls
	res += sizeof(dtype_t) * config->dim * config->n_heads * head_size * config->n_layers;
	res += sizeof(dtype_t) * config->dim * config->n_kv_heads * head_size * config->n_layers;
	res += sizeof(dtype_t) * config->dim * config->n_kv_heads * head_size * config->n_layers;
	res += sizeof(dtype_t) * config->dim * config->n_heads * head_size * config->n_layers;
	// weights for ffn
	res += sizeof(dtype_t) * config->hidden_dim * config->dim * config->n_layers;
	res += sizeof(dtype_t) * config->dim * config->hidden_dim * config->n_layers;
	res += sizeof(dtype_t) * config->hidden_dim * config->dim * config->n_layers;
	// final rmsnorm
	res += sizeof(dtype_t) * config->dim;
	// classifier weights for the logits, on the last layer
	res += sizeof(dtype_t) * config->vocab_size * config->dim;

	return res;
}

size_t kvcache_bandwidth(struct Config* config, int pos) {
	assert(pos < config->seq_len);

	int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;

	size_t res = 0;

	res += sizeof(kvtype_t) * config->n_layers * kv_dim * (pos + 1);
	res += sizeof(kvtype_t) * config->n_layers * kv_dim * (pos + 1);

	return res;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(struct Transformer* transformer, struct Tokenizer* tokenizer, struct Sampler* sampler, char* prompt, int steps) {
	char* empty_prompt = "";
	if (prompt == NULL) {
		prompt = empty_prompt;
	}

	// encode the (string) prompt into tokens sequence
	int* prompt_tokens = (int*)malloc(tokenizer_bound(strlen(prompt)) * sizeof(int));
	int num_prompt_tokens = tokenizer_encode(tokenizer, prompt, TF_ENCODE_BOS, prompt_tokens);
	if (num_prompt_tokens < 1) {
		fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
		exit(EXIT_FAILURE);
	}

	char* tokens_env = getenv("CALM_TOKENS");
	if (tokens_env && atoi(tokens_env)) {
		for (int i = 0; i < num_prompt_tokens; i++) {
			printf("[%s]", tokenizer_decode(tokenizer, prompt_tokens[i], prompt_tokens[i]));
		}
		printf("\n");
	}

	// hack for profiling: offset pos to make sure we need to use most of kv cache
	char* pos_offset_env = getenv("CALM_POSO");
	int pos_offset = pos_offset_env ? atoi(pos_offset_env) : 0;

	// start the main loop
	size_t read_bytes = 0;
	long start = time_in_ms();

	int next;                     // will store the next token in the sequence
	int token = prompt_tokens[0]; // kick off with the first token in the prompt
	int pos = 0;                  // position in the sequence

	while (pos < steps) {
		// forward the transformer to get logits for the next token
		unsigned flags = pos < num_prompt_tokens - 1 ? FF_UPDATE_KV_ONLY : 0;
		float* logits = transformer->forward(transformer, token, pos + pos_offset, flags);

		read_bytes += model_bandwidth(&transformer->config);
		read_bytes += kvcache_bandwidth(&transformer->config, pos + pos_offset);

		// advance the state machine
		if (pos < num_prompt_tokens - 1) {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos + 1];
		} else {
			// otherwise sample the next token from the logits
			next = sample(sampler, logits);
		}
		pos++;

		// data-dependent terminating condition: the BOS token delimits sequences, EOS token ends the sequence
		if (next == tokenizer->bos_id || next == tokenizer->eos_id) {
			break;
		}

		// print the token as string, decode it with the Tokenizer object
		char* piece = tokenizer_decode(tokenizer, token, next);
		printf("%s", piece);
		fflush(stdout);
		token = next;
	}
	printf("\n");

	long end = time_in_ms();
	fprintf(stderr, "# %d tokens: throughput: %.2f tok/s; latency: %.2f ms/tok; bandwidth: %.2f GB/s; total %.3f sec\n",
	        pos,
	        pos / (double)(end - start) * 1000, (double)(end - start) / pos,
	        ((double)read_bytes / 1e9) / ((double)(end - start) / 1000),
	        (double)(end - start) / 1000);

	free(prompt_tokens);
}

void study(struct Transformer* transformer, struct Tokenizer* tokenizer, const char* path) {
	int max_input_size = 32 * 1024;
	int max_tokens = tokenizer_bound(max_input_size);

	FILE* file = fopen(path, "r");
	if (!file) {
		fprintf(stderr, "failed to open %s\n", path);
		exit(EXIT_FAILURE);
	}

	char* input = (char*)malloc(max_input_size + 1);
	size_t input_size = fread(input, 1, max_input_size, file);
	fclose(file);

	input[input_size] = '\0';

	long start = time_in_ms();

	int* tokens = (int*)malloc(max_tokens * sizeof(int));
	int n_tokens = tokenizer_encode(tokenizer, input, TF_ENCODE_BOS, tokens);

	long mid = time_in_ms();

	free(input);

	printf("# %s: %d tokens (%.2f sec, %.2f tok/s)\n",
	       path, n_tokens, (double)(mid - start) / 1000, (double)n_tokens / (double)(mid - start) * 1000);

	int vocab_size = transformer->config.vocab_size;

	double sum = 0, den = 0;

	for (int i = 0; i + 1 < n_tokens; i++) {
		if (i != 0 && i % 1000 == 0) {
			printf("# progress (%d/%d): %.3f\n", i, n_tokens, exp(-sum / den));
		}

		// for now we reset the context after reaching the end of the window; this will result in artifically higher perplexity
		// note that this also means we don't get the BOS token which we might need to fix later...
		int pos = i % transformer->config.seq_len;
		float* logits = transformer->forward(transformer, tokens[i], pos, 0);

		sample_softmax(logits, vocab_size);

		float prob = logits[tokens[i + 1]];

		sum += log(prob);
		den += 1;
	}

	long end = time_in_ms();

	free(tokens);

	double ppl = exp(-sum / den);

	printf("# perplexity: %.3f (%.2f sec, %.2f tok/s)\n",
	       ppl, (double)(end - mid) / 1000, (double)(n_tokens - 1) / (double)(end - mid) * 1000);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
	fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
	fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
	fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
	fprintf(stderr, "  -r <int>    number of sequences to decode, default 1\n");
	fprintf(stderr, "  -i <string> input prompt\n");
	fprintf(stderr, "  -x <path>   compute perplexity for text file\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

	// default parameters
	char* checkpoint_path = NULL;    // e.g. out/model.bin
	float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
	int steps = 256;                 // number of steps to run for
	int sequences = 1;               // number of sequences to decode
	char* prompt = NULL;             // prompt string
	char* perplexity = NULL;         // text file for perplexity
	unsigned long long rng_seed = 0; // seed rng with time by default

	// poor man's C argparse so we can override the defaults above from the command line
	if (argc >= 2) {
		checkpoint_path = argv[1];
	} else {
		error_usage();
	}
	for (int i = 2; i < argc; i += 2) {
		// do some basic validation
		if (i + 1 >= argc) {
			error_usage();
		} // must have arg after flag
		if (argv[i][0] != '-') {
			error_usage();
		} // must start with dash
		if (strlen(argv[i]) != 2) {
			error_usage();
		} // must be -x (one dash, one letter)
		// read in the args
		if (argv[i][1] == 't') {
			temperature = atof(argv[i + 1]);
		} else if (argv[i][1] == 'p') {
			topp = atof(argv[i + 1]);
		} else if (argv[i][1] == 's') {
			rng_seed = atoi(argv[i + 1]);
		} else if (argv[i][1] == 'n') {
			steps = atoi(argv[i + 1]);
		} else if (argv[i][1] == 'r') {
			sequences = atoi(argv[i + 1]);
		} else if (argv[i][1] == 'i') {
			prompt = argv[i + 1];
		} else if (argv[i][1] == 'x') {
			perplexity = argv[i + 1];
		} else {
			error_usage();
		}
	}

	// parameter validation/overrides
	if (rng_seed <= 0)
		rng_seed = (unsigned int)time(NULL);
	if (temperature < 0.0)
		temperature = 0.0;
	if (topp < 0.0 || 1.0 < topp)
		topp = 0.9;
	if (steps < 0)
		steps = 0;

	// read .safetensors model
	struct Tensors tensors = {};
	if (tensors_open(&tensors, checkpoint_path) != 0) {
		fprintf(stderr, "failed to open tensors\n");
		exit(EXIT_FAILURE);
	}

	// build transformer using tensors from the input model file
	struct Transformer transformer = {};
	build_transformer(&transformer.config, &transformer.weights, &tensors);

	printf("# %s: %d layers, %d context, weights %.1f GiB (fp%d), KV cache %.1f GiB (fp%d)\n",
	       checkpoint_path, transformer.config.n_layers, transformer.config.seq_len,
	       (double)model_bandwidth(&transformer.config) / 1024 / 1024 / 1024,
	       (int)sizeof(dtype_t) * 8,
	       (double)kvcache_bandwidth(&transformer.config, transformer.config.seq_len - 1) / 1024 / 1024 / 1024,
	       (int)sizeof(kvtype_t) * 8);

	char* cpu = getenv("CALM_CPU");

	if (cpu && atoi(cpu)) {
		prepare(&transformer);
		transformer.forward = forward;
	} else {
		prepare_cuda(&transformer);
		transformer.forward = forward_cuda;
	}

	if (steps == 0 || steps > transformer.config.seq_len)
		steps = transformer.config.seq_len; // ovrerride to ~max length

	// build the Tokenizer via the tokenizer .bin file
	struct Tokenizer tokenizer;
	build_tokenizer(&tokenizer, &tensors, transformer.config.vocab_size);

	// build the Sampler
	struct Sampler sampler;
	sampler_init(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

	// do one inference as warmup
	// when using cpu, this makes sure tensors are loaded into memory (via mmap)
	// when using cuda, this makes sure all kernels are compiled and instantiated
	transformer.forward(&transformer, 0, 0, 0);
	profiler_reset();

	// run!
	if (perplexity) {
		study(&transformer, &tokenizer, perplexity);
	} else {
		for (int s = 0; s < sequences; ++s) {
			generate(&transformer, &tokenizer, &sampler, prompt, steps);
		}
	}

	profiler_dump();

	// memory and file handles cleanup
	// TODO: free transformer.state and transformer.weights for CUDA
	sampler_free(&sampler);
	tokenizer_free(&tokenizer);
	tensors_close(&tensors);
	return 0;
}
#endif
