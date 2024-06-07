// Inference for Llama-2 Transformer model in pure C
// Based on llama2.c by Andrej Karpathy

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "sampler.h"
#include "tensors.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// Transformer model

void prepare(struct Transformer* transformer);
float* forward(struct Transformer* transformer, int token, int pos, unsigned flags);

void* upload_cuda(void* host, size_t size);
void prepare_cuda(struct Transformer* transformer);
float* forward_cuda(struct Transformer* transformer, int token, int pos, unsigned flags);
void perf_cuda(void);

void init_metal(void);
void* upload_metal(void* host, size_t size);
void prepare_metal(struct Transformer* transformer);
float* forward_metal(struct Transformer* transformer, int token, int pos, unsigned flags);

void get_config(struct Config* config, struct Tensors* tensors, int context) {
	config->dim = atoi(tensors_metadata(tensors, "dim"));
	config->hidden_dim = atoi(tensors_metadata(tensors, "hidden_dim"));
	config->n_layers = atoi(tensors_metadata(tensors, "n_layers"));
	config->n_heads = atoi(tensors_metadata(tensors, "n_heads"));
	config->n_kv_heads = atoi(tensors_metadata(tensors, "n_kv_heads"));
	config->vocab_size = atoi(tensors_metadata(tensors, "vocab_size"));

	const char* head_dim = tensors_metadata_find(tensors, "head_dim");
	config->head_dim = head_dim ? atoi(head_dim) : config->dim / config->n_heads;

	// for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
	const char* max_seq_len = tensors_metadata_find(tensors, "max_seq_len");
	config->seq_len = max_seq_len && atoi(max_seq_len) < 4096 ? atoi(max_seq_len) : 4096;

	if (context) {
		config->seq_len = context;
	}

	config->rope_theta = atof(tensors_metadata(tensors, "rope_theta"));
	config->rotary_dim = atoi(tensors_metadata(tensors, "rotary_dim"));

	if (tensors_metadata_find(tensors, "n_experts")) {
		config->n_experts = atoi(tensors_metadata(tensors, "n_experts"));
		config->n_experts_ac = atoi(tensors_metadata(tensors, "n_experts_active"));
	}

	const char* norm_eps = tensors_metadata_find(tensors, "norm_eps");
	config->norm_eps = norm_eps ? atof(norm_eps) : 1e-5;

	const char* act_type = tensors_metadata_find(tensors, "act_type");
	config->act_gelu = act_type && strcmp(act_type, "gelu") == 0;

	const char* norm_type = tensors_metadata_find(tensors, "norm_type");
	config->norm_ln = norm_type && strncmp(norm_type, "layernorm", 9) == 0;  // note: we currently don't support layernorm bias
	config->norm_par = norm_type && strcmp(norm_type, "layernorm_par") == 0; // note: we currently don't support layernorm bias

	const char* qkv_clip = tensors_metadata_find(tensors, "qkv_clip");
	config->qkv_clip = qkv_clip ? atof(qkv_clip) : FLT_MAX;
}

void get_weights(struct Config* config, struct Weights* weights, struct Tensors* tensors) {
	const char* dtype = tensors_metadata(tensors, "dtype");

	enum DType wtype = strcmp(dtype, "gf4") == 0 ? dt_i32 : (strcmp(dtype, "fp8") == 0 ? dt_f8e5m2 : dt_f16);
	int gsize = strcmp(dtype, "gf4") == 0 ? 8 : 1;

	weights->dbits = strcmp(dtype, "gf4") == 0 ? 4 : (strcmp(dtype, "fp8") == 0 ? 8 : 16);

	weights->token_embedding_table = tensors_get(tensors, "model.embed.weight", 0, wtype, (int[]){config->vocab_size, config->dim / gsize, 0, 0});

	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (float*)tensors_get(tensors, "model.layers.%d.attn.norm.weight", l, dt_f32, (int[]){config->dim, 0, 0, 0});

		if (!config->norm_par) {
			weights->rms_ffn_weight[l] = (float*)tensors_get(tensors, "model.layers.%d.mlp.norm.weight", l, dt_f32, (int[]){config->dim, 0, 0, 0});
		}

		weights->wq[l] = tensors_get(tensors, "model.layers.%d.attn.wq.weight", l, wtype, (int[]){config->n_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wk[l] = tensors_get(tensors, "model.layers.%d.attn.wk.weight", l, wtype, (int[]){config->n_kv_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wv[l] = tensors_get(tensors, "model.layers.%d.attn.wv.weight", l, wtype, (int[]){config->n_kv_heads * config->head_dim, config->dim / gsize, 0, 0});
		weights->wo[l] = tensors_get(tensors, "model.layers.%d.attn.wo.weight", l, wtype, (int[]){config->dim, config->n_heads * config->head_dim / gsize, 0, 0});

		if (tensors_find(tensors, "model.layers.%d.attn.wqkv.bias", l)) {
			weights->bqkv[l] = (float*)tensors_get(tensors, "model.layers.%d.attn.wqkv.bias", l, dt_f32, (int[]){(config->n_heads + config->n_kv_heads * 2) * config->head_dim, 0, 0, 0});
		}

		if (config->n_experts) {
			weights->moegate[l] = tensors_get(tensors, "model.layers.%d.moegate.weight", l, wtype, (int[]){config->n_experts, config->dim / gsize, 0, 0});

			weights->w1[l] = tensors_get(tensors, "model.layers.%d.mlp.w1.weight", l, wtype, (int[]){config->n_experts, config->hidden_dim, config->dim / gsize, 0});
			weights->w2[l] = tensors_get(tensors, "model.layers.%d.mlp.w2.weight", l, wtype, (int[]){config->n_experts, config->dim, config->hidden_dim / gsize, 0});
			weights->w3[l] = tensors_get(tensors, "model.layers.%d.mlp.w3.weight", l, wtype, (int[]){config->n_experts, config->hidden_dim, config->dim / gsize, 0});
		} else {
			weights->w1[l] = tensors_get(tensors, "model.layers.%d.mlp.w1.weight", l, wtype, (int[]){config->hidden_dim, config->dim / gsize, 0, 0});
			weights->w2[l] = tensors_get(tensors, "model.layers.%d.mlp.w2.weight", l, wtype, (int[]){config->dim, config->hidden_dim / gsize, 0, 0});
			weights->w3[l] = tensors_get(tensors, "model.layers.%d.mlp.w3.weight", l, wtype, (int[]){config->hidden_dim, config->dim / gsize, 0, 0});
		}
	}

	weights->rms_final_weight = (float*)tensors_get(tensors, "model.norm.weight", 0, dt_f32, (int[]){config->dim, 0, 0, 0});

	if (tensors_find(tensors, "model.output.weight", 0) == NULL) {
		weights->wcls = weights->token_embedding_table; // tied weights
	} else {
		weights->wcls = tensors_get(tensors, "model.output.weight", 0, wtype, (int[]){config->vocab_size, config->dim / gsize, 0, 0});
	}
}

void build_tokenizer(struct Tokenizer* t, struct Tensors* tensors, int vocab_size) {
	struct Tensor* tensor = tensors_find(tensors, "tokenizer.tokens", 0);

	char* tokens = (char*)tensors_get(tensors, "tokenizer.tokens", 0, dt_u8, (int[]){tensor->shape[0], 0, 0, 0});
	float* scores = (float*)tensors_get(tensors, "tokenizer.scores", 0, dt_f32, (int[]){vocab_size, 0, 0, 0});

	int bos_id = atoi(tensors_metadata(tensors, "bos_token_id"));
	int eos_id = atoi(tensors_metadata(tensors, "eos_token_id"));

	tokenizer_init(t, tokens, scores, bos_id, eos_id, vocab_size, tensor->shape[0]);
}

size_t count_bytes(struct Tensors* tensors, const char* prefix, const char* filter, size_t* out_params) {
	size_t bytes = 0, params = 0;
	for (int i = 0; i < tensors->n_tensors; ++i) {
		struct Tensor* tensor = &tensors->tensors[i];
		if (strncmp(tensor->name, prefix, strlen(prefix)) != 0) {
			continue;
		}
		if (filter && strstr(tensor->name, filter) == NULL) {
			continue;
		}
		int elts = tensor->dtype == dt_i32 ? 8 : 1; // gsize hack for gf4
		for (int j = 0; j < 4 && tensor->shape[j] != 0; ++j) {
			elts *= tensor->shape[j];
		}
		params += elts;
		bytes += tensor->size;
	}
	if (out_params) {
		*out_params = params;
	}
	return bytes;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
	// return time in milliseconds, for benchmarking the model speed
	struct timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

size_t kvcache_bandwidth(struct Config* config, int kvbits, int pos) {
	int kv_dim = config->head_dim * config->n_kv_heads;
	int kv_len = pos >= config->seq_len ? config->seq_len : pos + 1;
	return 2 * (size_t)(kvbits / 8) * config->n_layers * kv_dim * kv_len;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(struct Transformer* transformer, struct Tokenizer* tokenizer, struct Sampler* sampler, char* prompt, int steps, int pos_offset) {
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
			printf("[%s:%d]", tokenizer_decode(tokenizer, prompt_tokens[i], prompt_tokens[i]), prompt_tokens[i]);
		}
		printf("\n");
	}

	// start the main loop
	size_t read_bytes = 0;
	long start = time_in_ms();

	int next;                     // will store the next token in the sequence
	int token = prompt_tokens[0]; // kick off with the first token in the prompt
	int pos = 0;                  // position in the sequence

	// print first prompt token since it won't be decoded
	if (token != tokenizer->bos_id) {
		char* piece = tokenizer_decode(tokenizer, tokenizer->bos_id, token);
		printf("%s", piece);
		fflush(stdout);
	}

	float* logits_last = NULL;

	while (pos < steps || steps < 0) {
		// forward the transformer to get logits for the next token
		unsigned flags = pos < num_prompt_tokens - 1 ? FF_UPDATE_KV_ONLY : 0;
		float* logits = transformer->forward(transformer, token, pos + pos_offset, flags);

		read_bytes += transformer->n_bandwidth;
		read_bytes += kvcache_bandwidth(&transformer->config, transformer->state.kvbits, pos + pos_offset);
		logits_last = logits;

		// advance the state machine
		if (pos < num_prompt_tokens - 1) {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos + 1];
		} else {
			// otherwise sample the next token from the logits
			next = sample(sampler, logits);
			assert(next >= 0);

			// data-dependent terminating condition: the BOS token delimits sequences, EOS token ends the sequence, EOT token ends the turn
			if (next == tokenizer->bos_id || next == tokenizer->eos_id || next == tokenizer->eot_id) {
				break;
			}
		}
		pos++;

		// print the token as string, decode it with the Tokenizer object
		char* piece = tokenizer_decode(tokenizer, token, next);
		printf("%s", piece);
		fflush(stdout);
		token = next;
	}
	printf("\n");

	// fold last token's logits into a hash for validation
	unsigned logits_hash = 0;
	for (int k = 0; k < transformer->config.vocab_size; ++k) {
		logits_hash = logits_hash * 5 + *(unsigned*)(&logits_last[k]);
	}

	long end = time_in_ms();
	fprintf(stderr, "# %d tokens: throughput: %.2f tok/s; latency: %.2f ms/tok; bandwidth: %.2f GB/s; total %.3f sec; #%08x\n",
	        pos,
	        pos / (double)(end - start) * 1000, (double)(end - start) / pos,
	        ((double)read_bytes / 1e9) / ((double)(end - start) / 1000),
	        (double)(end - start) / 1000, logits_hash);

	free(prompt_tokens);
}

void study(struct Transformer* transformer, struct Tokenizer* tokenizer, const char* path, int steps) {
	int max_input_size = 64 * 1024;
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

	printf("# %s: %d tokens (%.3f sec), chunked with size %d\n",
	       path, n_tokens, (double)(mid - start) / 1000, steps);

	int vocab_size = transformer->config.vocab_size;

	double sum = 0, ss = 0, den = 0;
	double ppl = 0, pplerr = 0;

	for (int i = 0; i + 1 < n_tokens; i++) {
		if (i != 0 && i % 1000 == 0) {
			printf("# progress (%d/%d): %.3f ± %.3f\n", i, n_tokens, ppl, pplerr);
		}

		int pos = steps <= 0 ? i : i % steps;
		float* logits = transformer->forward(transformer, tokens[i], pos, 0);
		double logprob = log(sample_prob(tokens[i + 1], logits, vocab_size));

		// update stats for mean/std
		sum += logprob;
		ss += logprob * logprob;
		den += 1;

		// update ppl and ppl error using standard error of the mean
		ppl = exp(-sum / den);
		pplerr = ppl * sqrt((ss - sum * sum / den) / den / den);
	}

	long end = time_in_ms();

	free(tokens);

	printf("# perplexity: %.3f ± %.3f (%.2f sec, %.2f tok/s)\n",
	       ppl, pplerr, (double)(end - mid) / 1000, (double)(n_tokens - 1) / (double)(end - mid) * 1000);
}

static const char* chatframe(struct Tokenizer* tokenizer, bool has_system) {
	if (tokenizer_find(tokenizer, "<|eot_id|>") >= 0) {
		// llama3
		return has_system ? "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
		                  : "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
	} else if (tokenizer_find(tokenizer, "<|im_start|>") >= 0) {
		// chatml
		return has_system ? "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
		                  : "\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
	} else if (tokenizer_find(tokenizer, "<start_of_turn>") >= 0) {
		// gemma
		return has_system ? "<start_of_turn>user\nSYSTEM: %s\n\n%s<end_of_turn>\n<start_of_turn>model\n"
		                  : "\n<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n";
	} else if (tokenizer_find(tokenizer, "<|START_OF_TURN_TOKEN|>") >= 0) {
		// cohere
		return has_system ? "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>%s<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>%s<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
		                  : "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>%s<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
	} else if (tokenizer_find(tokenizer, "<|assistant|>") >= 0) {
		// phi3
		return has_system ? "<|system|>\n%s<|end|>\n<|user|>\n%s<|end|>\n<|assistant|>\n"
		                  : "\n<|user|>\n%s<|end|>\n<|assistant|>\n";
	} else if (tokenizer_find(tokenizer, "<|beginofsystem|>") >= 0) {
		// k2
		return has_system ? "<|beginofsystem|>%s<|endofsystemprompt|><|beginofuser|>%s<|beginofsystem|>"
		                  : "<|beginofuser|>%s<|beginofsystem|>";
	} else {
		// llama
		return has_system ? "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]" : "[INST] %s [/INST]";
	}
}

void chat(struct Transformer* transformer, struct Tokenizer* tokenizer, struct Sampler* sampler, char* cli_prompt, char* system_prompt) {
	char user_prompt[512];
	char rendered_prompt[sizeof(user_prompt) * 2];
	int prompt_tokens[sizeof(rendered_prompt) + 4];
	int num_prompt_tokens = 0;

	int user_idx = 0;
	int user_turn = 1; // user starts
	int next = 0;      // will store the next token in the sequence
	int token = 0;     // stores the current token to feed into the transformer
	int pos = 0;       // position in the sequence
	for (;;) {
		// when it is the user's turn to contribute tokens to the dialog...
		if (user_turn) {
			// get the user prompt
			if (pos == 0 && cli_prompt != NULL) {
				// user prompt for position 0 was passed in, use it
				snprintf(user_prompt, sizeof(user_prompt), "%s\n", cli_prompt);
			} else {
				// otherwise get user prompt from stdin
				double seq_pct = (double)pos / (double)transformer->config.seq_len;
				char progress[64] = {};
				for (int i = 0; i < 10; ++i) {
					strcat(progress, seq_pct < i * 0.1 ? "░" : (seq_pct < i * 0.1 + 0.05 ? "▒" : "█"));
				}
				printf("%s \033[32mUser:\033[37m ", progress);
				fflush(stdout);
				char* x = fgets(user_prompt, sizeof(user_prompt), stdin);
				(void)x;
			}
			// render user/system prompts into the chat schema
			if (pos == 0 && system_prompt[0] != '\0') {
				snprintf(rendered_prompt, sizeof(rendered_prompt), chatframe(tokenizer, true), system_prompt, user_prompt);
			} else {
				snprintf(rendered_prompt, sizeof(rendered_prompt), chatframe(tokenizer, false), user_prompt);
			}

			// encode the rendered prompt into tokens
			num_prompt_tokens = tokenizer_encode(tokenizer, rendered_prompt, pos == 0 ? TF_ENCODE_BOS : 0, prompt_tokens);
			user_idx = 0; // reset the user index
			user_turn = 0;
			printf("\n\033[33mAssistant:\033[00m ");
		}

		// determine the token to pass into the transformer next
		if (user_idx < num_prompt_tokens) {
			// if we are still processing the input prompt, force the next prompt token
			token = prompt_tokens[user_idx++];
		} else {
			// otherwise use the next token sampled from previous turn
			token = next;
		}

		// forward the transformer to get logits for the next token
		unsigned flags = user_idx < num_prompt_tokens ? FF_UPDATE_KV_ONLY : 0;
		float* logits = transformer->forward(transformer, token, pos, flags);
		pos++;

		if (user_idx >= num_prompt_tokens) {
			next = sample(sampler, logits);

			if (next == tokenizer->eos_id || next == tokenizer->eot_id) {
				// EOS token ends the Assistant turn
				printf("\n\n");
				user_turn = 1;
			} else {
				// the Assistant is responding, so print its output
				char* piece = tokenizer_decode(tokenizer, token, next);
				printf("%s", piece);
				fflush(stdout);
			}
		}
	}
}

void error_usage() {
	fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
	fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
	fprintf(stderr, "  -p <float>  p value in min-p (cutoff) sampling in [0,1] default 0.1\n");
	fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
	fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len, -1 = infinite\n");
	fprintf(stderr, "  -r <int>    number of sequences to decode, default 1\n");
	fprintf(stderr, "  -c <int>    context length, default to model-specific maximum\n");
	fprintf(stderr, "  -i <string> input prompt (- to read from stdin)\n");
	fprintf(stderr, "  -x <path>   compute perplexity for text file\n");
	fprintf(stderr, "  -y <string> chat mode with a system prompt\n");
	exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {

	// default parameters
	char* checkpoint_path = NULL;    // e.g. out/model.bin
	float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
	float minp = 0.1f;               // min-p sampling. 0.0 = off
	int steps = 256;                 // number of steps to run for
	int sequences = 1;               // number of sequences to decode
	char* prompt = NULL;             // prompt string
	char* perplexity = NULL;         // text file for perplexity
	char* system_prompt = NULL;      // chat system prompt
	unsigned long long rng_seed = 0; // seed rng with time by default
	int context = 0;                 // context length

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
			minp = atof(argv[i + 1]);
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
		} else if (argv[i][1] == 'c') {
			context = atoi(argv[i + 1]);
		} else if (argv[i][1] == 'y') {
			system_prompt = argv[i + 1];
		} else {
			error_usage();
		}
	}

	// parameter validation/overrides
	if (rng_seed <= 0)
		rng_seed = (unsigned int)time(NULL);

	if (prompt && strcmp(prompt, "-") == 0) {
		static char input[65536];
		size_t input_size = fread(input, 1, sizeof(input) - 1, stdin);
		input[input_size] = '\0';
		prompt = input;
	}

#ifdef __linux__
	char* cpu = getenv("CALM_CPU");
	bool cuda = !cpu || atoi(cpu) == 0;
#endif

#ifdef __APPLE__
	char* cpu = getenv("CALM_CPU");
	bool metal = !cpu || atoi(cpu) == 0;
#endif

	// read .safetensors model
	struct Tensors tensors = {};
	if (tensors_open(&tensors, checkpoint_path) != 0) {
		fprintf(stderr, "failed to open %s\n", checkpoint_path);
		exit(EXIT_FAILURE);
	}

	// build transformer using tensors from the input model file
	struct Transformer transformer = {};
	get_config(&transformer.config, &tensors, context);
	transformer.n_bytes = count_bytes(&tensors, "model.", NULL, &transformer.n_params);
	transformer.n_bandwidth = transformer.n_bytes - count_bytes(&tensors, "model.embed.", NULL, NULL);
	if (tensors_find(&tensors, "model.output.weight", 0) == NULL) {
		transformer.n_bandwidth += tensors_find(&tensors, "model.embed.weight", 0)->size;
	}
	if (transformer.config.n_experts) {
		size_t mlp = count_bytes(&tensors, "model.layers.", ".mlp.w", NULL);
		transformer.n_bandwidth -= mlp;
		transformer.n_bandwidth += mlp / transformer.config.n_experts * transformer.config.n_experts_ac;
	}

	transformer.state.kvbits = 16;

#ifdef __linux__
	if (cuda && transformer.config.seq_len > 4096) {
		transformer.state.kvbits = 8; // for now use fp8 for larger contexts automatically without explicit control
	}
#endif

	printf("# %s: %.1fB params (%.1f GiB @ %.2f bpw), %d context (kvcache %.1f GiB @ fp%d)\n",
	       checkpoint_path,
	       (double)transformer.n_params / 1e9, (double)transformer.n_bytes / 1024 / 1024 / 1024,
	       (double)transformer.n_bytes * 8 / (double)transformer.n_params,
	       transformer.config.seq_len,
	       (double)kvcache_bandwidth(&transformer.config, transformer.state.kvbits, transformer.config.seq_len - 1) / 1024 / 1024 / 1024,
	       transformer.state.kvbits);

#ifdef __linux__
	// upload tensors to the GPU
	if (cuda) {
		int i;
		for (i = 0; i < tensors.n_tensors; ++i) {
			struct Tensor* tensor = &tensors.tensors[i];
			if (strncmp(tensor->name, "model.", 6) == 0) {
				tensor->data = upload_cuda(tensor->data, tensor->size);
			}
		}
	}
#endif

#ifdef __APPLE__
	// upload tensors to the GPU
	if (metal) {
		init_metal();
		for (int i = 0; i < tensors.n_tensors; ++i) {
			struct Tensor* tensor = &tensors.tensors[i];
			if (strncmp(tensor->name, "model.", 6) == 0) {
				tensor->data = upload_metal(tensor->data, tensor->size);
			}
		}
	}
#endif

	get_weights(&transformer.config, &transformer.weights, &tensors);

#ifdef __linux__
	if (cuda) {
		prepare_cuda(&transformer);
		transformer.forward = forward_cuda;
	}
#endif

#ifdef __APPLE__
	if (metal) {
		prepare_metal(&transformer);
		transformer.forward = forward_metal;
	}
#endif

	// CPU fallback
	if (!transformer.forward) {
		prepare(&transformer);
		transformer.forward = forward;
	}

	// build the Tokenizer via the tokenizer .bin file
	struct Tokenizer tokenizer;
	build_tokenizer(&tokenizer, &tensors, transformer.config.vocab_size);

	// build the Sampler
	struct Sampler sampler = {transformer.config.vocab_size, rng_seed, temperature, minp};

	// hack for profiling: offset pos to make sure we need to use most of kv cache
	char* pos_offset_env = getenv("CALM_POSO");
	int pos_offset = pos_offset_env ? atoi(pos_offset_env) : 0;

	// do one inference as warmup
	// when using cpu, this makes sure tensors are loaded into memory (via mmap)
	// when using cuda, this makes sure all kernels are compiled and instantiated
	transformer.forward(&transformer, 0, pos_offset, 0);

	// -n 0 means use the full context length
	if (steps == 0)
		steps = transformer.config.seq_len;

	// run!
	if (perplexity) {
		study(&transformer, &tokenizer, perplexity, steps);
	} else if (system_prompt) {
		chat(&transformer, &tokenizer, &sampler, prompt, system_prompt);
	} else {
		for (int s = 0; s < sequences; ++s) {
			generate(&transformer, &tokenizer, &sampler, prompt, steps, pos_offset);
		}
	}

#ifdef __linux__
	if (cuda && getenv("CUDA_INJECTION64_PATH")) {
		perf_cuda();
	}
#endif

	// memory and file handles cleanup
	// TODO: free transformer.state and transformer.weights for CUDA
	tokenizer_free(&tokenizer);
	tensors_close(&tensors);
	return 0;
}
