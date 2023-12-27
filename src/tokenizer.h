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

enum TokenizerFlags {
	TF_ENCODE_BOS = 1 << 0,
	TF_ENCODE_EOS = 1 << 1,
};

void tokenizer_init(struct Tokenizer* tokenizer, char* tokens, float* scores, int bos_id, int eos_id, int vocab_size);
void tokenizer_free(struct Tokenizer* tokenizer);

int tokenizer_bound(int bytes);

char* tokenizer_decode(struct Tokenizer* tokenizer, int prev_token, int token);
int tokenizer_encode(struct Tokenizer* tokenizer, char* text, unsigned flags, int* tokens);
