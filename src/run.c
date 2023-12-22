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
#include "tensors.h"

// ----------------------------------------------------------------------------
// Transformer model

void prepare(struct Transformer* transformer);
float* forward(struct Transformer* transformer, int token, int pos);

void prepare_cuda(struct Transformer* transformer);
float* forward_cuda(struct Transformer* transformer, int token, int pos);

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

	weights->token_embedding_table = (dtype_t*)tensors_get(tensors, "model.embed_tokens.weight", 0, dtype, (int[]){config->vocab_size, config->dim, 0, 0});
	for (int l = 0; l < config->n_layers; ++l) {
		weights->rms_att_weight[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.input_layernorm.weight", l, dtype, (int[]){config->dim, 0, 0, 0});
		weights->wq[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.self_attn.q_proj.weight", l, dtype, (int[]){config->dim, config->n_heads * head_size, 0, 0});
		weights->wk[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.self_attn.k_proj.weight", l, dtype, (int[]){config->n_kv_heads * head_size, config->dim, 0, 0});
		weights->wv[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.self_attn.v_proj.weight", l, dtype, (int[]){config->n_kv_heads * head_size, config->dim, 0, 0});
		weights->wo[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.self_attn.o_proj.weight", l, dtype, (int[]){config->n_heads * head_size, config->dim, 0, 0});
		weights->rms_ffn_weight[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.post_attention_layernorm.weight", l, dtype, (int[]){config->dim, 0, 0, 0});
		weights->w1[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.gate_proj.weight", l, dtype, (int[]){config->hidden_dim, config->dim, 0, 0});
		weights->w2[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.down_proj.weight", l, dtype, (int[]){config->dim, config->hidden_dim, 0, 0});
		weights->w3[l] = (dtype_t*)tensors_get(tensors, "model.layers.%d.mlp.up_proj.weight", l, dtype, (int[]){config->hidden_dim, config->dim, 0, 0});
	}
	weights->rms_final_weight = (dtype_t*)tensors_get(tensors, "model.norm.weight", 0, dtype, (int[]){config->dim, 0, 0, 0});
	weights->wcls = (dtype_t*)tensors_get(tensors, "lm_head.weight", 0, dtype, (int[]){config->vocab_size, config->dim, 0, 0});
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

#define MAX_TOKEN_LENGTH 128

typedef struct {
	char* str;
	int id;
} TokenIndex;

typedef struct {
	char** vocab;
	float* vocab_scores;
	TokenIndex* sorted_vocab;
	int vocab_size;
	unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void* a, const void* b) {
	return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, struct Tensors* tensors, int vocab_size) {
	t->vocab_size = vocab_size;

	struct Tensor* tensor = tensors_find(tensors, "tokenizer.tokens", 0);

	char* tokens = (char*)tensors_get(tensors, "tokenizer.tokens", 0, dt_u8, (int[]){tensor->shape[0], 0, 0, 0});
	float* scores = (float*)tensors_get(tensors, "tokenizer.scores", 0, dt_f32, (int[]){vocab_size, 0, 0, 0});

	// TODO: this is likely redundant
	for (int i = 0; i < 256; i++) {
		t->byte_pieces[i * 2] = (unsigned char)i;
		t->byte_pieces[i * 2 + 1] = '\0';
	}

	// malloc space to hold the scores and the strings
	t->vocab = (char**)malloc(vocab_size * sizeof(char*));
	t->sorted_vocab = (TokenIndex*)malloc(vocab_size * sizeof(TokenIndex));

	// TODO: validate tokens are null terminated
	for (int i = 0; i < vocab_size; ++i) {
		t->vocab[i] = tokens;
		t->sorted_vocab[i].str = tokens;
		t->sorted_vocab[i].id = i;

		assert(strlen(tokens) < MAX_TOKEN_LENGTH);
		tokens += strlen(tokens) + 1;
	}

	t->vocab_scores = scores;

	qsort(t->sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
}

void free_tokenizer(Tokenizer* t) {
	free(t->vocab);
	free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
	char* piece = t->vocab[token];
	// following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
	if (prev_token == 1 && piece[0] == ' ') {
		piece++;
	}
	// careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
	// parse this and convert and return the actual byte
	unsigned char byte_val;
	if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
		piece = (char*)t->byte_pieces + byte_val * 2;
	}
	return piece;
}

void safe_printf(char* piece) {
	// piece might be a raw byte token, and we only want to print printable chars or whitespace
	// because some of the other bytes can be various control codes, backspace, etc.
	if (piece == NULL) {
		return;
	}
	if (piece[0] == '\0') {
		return;
	}
	if (piece[1] == '\0') {
		unsigned char byte_val = piece[0];
		if (!(isprint(byte_val) || isspace(byte_val))) {
			return; // bad byte, don't print it
		}
	}
	printf("%s", piece);
}

int str_lookup(char* str, TokenIndex* sorted_vocab, int vocab_size) {
	// efficiently find the perfect match for str in vocab, return its index or -1 if not found
	TokenIndex tok = {.str = str}; // acts as the key to search for
	TokenIndex* res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
	return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
	// encode the string text (input) into an upper-bound preallocated tokens[] array
	// bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
	if (text == NULL) {
		fprintf(stderr, "cannot encode NULL text\n");
		exit(EXIT_FAILURE);
	}

	// create a temporary buffer that will store merge candidates of always two consecutive tokens
	// *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
	char* str_buffer = malloc((MAX_TOKEN_LENGTH * 2 + 1 + 2) * sizeof(char));
	size_t str_len = 0;

	// start at 0 tokens
	*n_tokens = 0;

	// add optional BOS (=1) token, if desired
	if (bos)
		tokens[(*n_tokens)++] = 1;

	// add_dummy_prefix is true by default
	// so prepend a dummy prefix token to the input string, but only if text != ""
	// TODO: pretty sure this isn't correct in the general case but I don't have the
	// energy to read more of the sentencepiece code to figure out what it's doing
	if (text[0] != '\0') {
		int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
		tokens[(*n_tokens)++] = dummy_prefix;
	}

	// Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
	// Code point ↔ UTF-8 conversion
	// First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
	// U+0000	U+007F	    0xxxxxxx
	// U+0080	U+07FF	    110xxxxx	10xxxxxx
	// U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
	// U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

	// process the raw (UTF-8) byte sequence of the input string
	for (char* c = text; *c != '\0'; c++) {

		// reset buffer if the current byte is ASCII or a leading byte
		// 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
		// 0x80 is 10000000
		// in UTF-8, all continuation bytes start with "10" in first two bits
		// so in English this is: "if this byte is not a continuation byte"
		if ((*c & 0xC0) != 0x80) {
			// this byte must be either a leading byte (11...) or an ASCII char (0x...)
			// => reset our location, as we're starting a new UTF-8 codepoint
			str_len = 0;
		}

		// append the current byte to the buffer
		str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
		str_buffer[str_len] = '\0';

		// while the next character is a continuation byte, continue appending
		// but if there are too many of them, just stop to avoid overruning str_buffer size.
		if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
			continue;
		}

		// ok c+1 is not a continuation byte, so we've read in a full codepoint
		int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

		if (id != -1) {
			// we found this codepoint in vocab, add it as a token
			tokens[(*n_tokens)++] = id;
		} else {
			// byte_fallback encoding: just encode each byte as a token
			// +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
			// so the individual bytes only start at index 3
			for (int i = 0; i < str_len; i++) {
				tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
			}
		}
		str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
	}

	// merge the best consecutive pair each iteration, according the scores in vocab_scores
	while (1) {
		float best_score = -1e10;
		int best_id = -1;
		int best_idx = -1;

		for (int i = 0; i < (*n_tokens - 1); i++) {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
			int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
			if (id != -1 && t->vocab_scores[id] > best_score) {
				// this merge pair exists in vocab! record its score and position
				best_score = t->vocab_scores[id];
				best_id = id;
				best_idx = i;
			}
		}

		if (best_idx == -1) {
			break; // we couldn't find any more pairs to merge, so we're done
		}

		// merge the consecutive pair (best_idx, best_idx+1) into new token best_id
		tokens[best_idx] = best_id;
		// delete token at position best_idx+1, shift the entire sequence back 1
		for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
			tokens[i] = tokens[i + 1];
		}
		(*n_tokens)--; // token length decreased
	}

	// add optional EOS (=2) token, if desired
	if (eos)
		tokens[(*n_tokens)++] = 2;

	free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
	float prob;
	int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
	int vocab_size;
	ProbIndex* probindex; // buffer used in top-p sampling
	float temperature;
	float topp;
	unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
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

int sample_mult(float* probabilities, int n, float coin) {
	// sample index from probabilities (they must sum to 1!)
	// coin is a random number in [0, 1), usually from random_f32()
	float cdf = 0.0f;
	for (int i = 0; i < n; i++) {
		cdf += probabilities[i];
		if (coin < cdf) {
			return i;
		}
	}
	return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
	ProbIndex* a_ = (ProbIndex*)a;
	ProbIndex* b_ = (ProbIndex*)b;
	if (a_->prob > b_->prob)
		return -1;
	if (a_->prob < b_->prob)
		return 1;
	return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
	// top-p sampling (or "nucleus sampling") samples from the smallest set of
	// tokens that exceed probability topp. This way we never sample tokens that
	// have very low probabilities and are less likely to go "off the rails".
	// coin is a random number in [0, 1), usually from random_f32()

	int n0 = 0;
	// quicksort indices in descending order of probabilities
	// values smaller than (1 - topp) / (n - 1) cannot be part of the result
	// so for efficiency we crop these out as candidates before sorting
	const float cutoff = (1.0f - topp) / (n - 1);
	for (int i = 0; i < n; i++) {
		if (probabilities[i] >= cutoff) {
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			n0++;
		}
	}
	qsort(probindex, n0, sizeof(ProbIndex), compare);

	// truncate the list where cumulative probability exceeds topp
	float cumulative_prob = 0.0f;
	int last_idx = n0 - 1; // in case of rounding errors consider all elements
	for (int i = 0; i < n0; i++) {
		cumulative_prob += probindex[i].prob;
		if (cumulative_prob > topp) {
			last_idx = i;
			break; // we've exceeded topp by including last_idx
		}
	}

	// sample from the truncated list
	float r = coin * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i <= last_idx; i++) {
		cdf += probindex[i].prob;
		if (r < cdf) {
			return probindex[i].index;
		}
	}
	return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
	sampler->vocab_size = vocab_size;
	sampler->temperature = temperature;
	sampler->topp = topp;
	sampler->rng_state = rng_seed;
	// buffer only used with nucleus sampling; may not need but it's ~small
	sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
	free(sampler->probindex);
}

unsigned int random_u32(unsigned long long* state) {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;
	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) { // random float32 in [0,1)
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

int sample(Sampler* sampler, float* logits) {
	// sample the token given the logits and some hyperparameters
	int next;
	if (sampler->temperature == 0.0f) {
		// greedy argmax sampling: take the token with the highest probability
		next = sample_argmax(logits, sampler->vocab_size);
	} else {
		// apply the temperature to the logits
		for (int q = 0; q < sampler->vocab_size; q++) {
			logits[q] /= sampler->temperature;
		}
		// apply softmax to the logits to get the probabilities for next token
		sample_softmax(logits, sampler->vocab_size);
		// flip a (float) coin (this is our source of entropy for sampling)
		float coin = random_f32(&sampler->rng_state);
		// we sample from this distribution to get the next token
		if (sampler->topp <= 0 || sampler->topp >= 1) {
			// simply sample from the predicted probability distribution
			next = sample_mult(logits, sampler->vocab_size, coin);
		} else {
			// top-p (nucleus) sampling, clamping the least likely tokens to zero
			next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
		}
	}
	return next;
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

void generate(struct Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
	char* empty_prompt = "";
	if (prompt == NULL) {
		prompt = empty_prompt;
	}

	// encode the (string) prompt into tokens sequence
	int num_prompt_tokens = 0;
	int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
	encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
	if (num_prompt_tokens < 1) {
		fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
		exit(EXIT_FAILURE);
	}

	// start the main loop
	size_t read_bytes = 0;
	long start = 0;               // used to time our code, only initialized after first iteration
	int next;                     // will store the next token in the sequence
	int token = prompt_tokens[0]; // kick off with the first token in the prompt
	int pos = 0;                  // position in the sequence
	while (pos < steps) {

		// forward the transformer to get logits for the next token
		float* logits = transformer->forward(transformer, token, pos);

		read_bytes += model_bandwidth(&transformer->config);
		read_bytes += kvcache_bandwidth(&transformer->config, pos);

		// advance the state machine
		if (pos < num_prompt_tokens - 1) {
			// if we are still processing the input prompt, force the next prompt token
			next = prompt_tokens[pos + 1];
		} else {
			// otherwise sample the next token from the logits
			next = sample(sampler, logits);
		}
		pos++;

		// data-dependent terminating condition: the BOS (=1) token delimits sequences, EOS (=2) token ends the sequence
		if (next == 1 || next == 2) {
			break;
		}

		// print the token as string, decode it with the Tokenizer object
		char* piece = decode(tokenizer, token, next);
		safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
		fflush(stdout);
		token = next;

		// init the timer here because the first iteration can be slower
		if (start == 0) {
			start = time_in_ms();
		}
	}
	printf("\n");

	// report achieved tok/s (pos-1 because the timer starts after first iteration)
	if (pos > 1) {
		long end = time_in_ms();
		fprintf(stderr, "# %d tokens: throughput: %.2f tok/s; latency: %.2f ms/tok; bandwidth: %.2f GB/s\n",
		        pos,
		        (pos - 1) / (double)(end - start) * 1000,
		        (double)(end - start) / (pos - 1),
		        ((double)read_bytes / 1e9) / ((double)(end - start) / 1000));
	}

	free(prompt_tokens);
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
	fprintf(stderr, "  -a <string> use accelerator (cpu, cuda)\n");
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
	unsigned long long rng_seed = 0; // seed rng with time by default
	char* accelerator = "cpu";       // cpu|cuda

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
		} else if (argv[i][1] == 'a') {
			accelerator = argv[i + 1];
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

	printf("# %s: %d layers, %d context, weights %.1f GiB, KV cache %.1f GiB\n",
	       checkpoint_path, transformer.config.n_layers, transformer.config.seq_len,
	       (double)model_bandwidth(&transformer.config) / 1024 / 1024 / 1024,
	       (double)kvcache_bandwidth(&transformer.config, transformer.config.seq_len - 1) / 1024 / 1024 / 1024);

	if (strcmp(accelerator, "cuda") == 0) {
		prepare_cuda(&transformer);
		transformer.forward = forward_cuda;
	} else if (strcmp(accelerator, "cpu") == 0) {
		prepare(&transformer);
		transformer.forward = forward;
	} else {
		fprintf(stderr, "unknown accelerator: %s\n", accelerator);
		exit(EXIT_FAILURE);
	}

	if (steps == 0 || steps > transformer.config.seq_len)
		steps = transformer.config.seq_len; // ovrerride to ~max length

	// build the Tokenizer via the tokenizer .bin file
	Tokenizer tokenizer;
	build_tokenizer(&tokenizer, &tensors, transformer.config.vocab_size);

	// build the Sampler
	Sampler sampler;
	build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

	// run!
	for (int s = 0; s < sequences; ++s) {
		generate(&transformer, &tokenizer, &sampler, prompt, steps);
	}

	// memory and file handles cleanup
	// TODO: free transformer.state and transformer.weights for CUDA
	free_sampler(&sampler);
	free_tokenizer(&tokenizer);
	tensors_close(&tensors);
	return 0;
}
#endif
