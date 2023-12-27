#pragma once

struct TokenIndex {
	char* str;
	int id;
};

struct Tokenizer {
	char** vocab;
	float* vocab_scores;
	struct TokenIndex* sorted_vocab;

	int vocab_size;
	int bos_id;
	int eos_id;
	int byte_fallbacks;

	char byte_pieces[256][2];
};

void tokenizer_init(struct Tokenizer* tokenizer, char* tokens, float* scores, int bos_id, int eos_id, int vocab_size);
void tokenizer_free(struct Tokenizer* tokenizer);

char* tokenizer_decode(struct Tokenizer* tokenizer, int prev_token, int token);
void tokenizer_encode(struct Tokenizer* tokenizer, char* text, int bos, int eos, int* tokens, int* n_tokens);
