#include "tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKEN_LENGTH 128

static int compare_tokens(const void* a, const void* b) {
	return strcmp(((struct TokenIndex*)a)->str, ((struct TokenIndex*)b)->str);
}

static int str_lookup(char* str, struct TokenIndex* sorted_vocab, int vocab_size) {
	// efficiently find the perfect match for str in vocab, return its index or -1 if not found
	struct TokenIndex tok = {str, -1}; // acts as the key to search for
	struct TokenIndex* res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(struct TokenIndex), compare_tokens);
	return res != NULL ? res->id : -1;
}

void tokenizer_init(struct Tokenizer* tokenizer, char* tokens, float* scores, int bos_id, int eos_id, int vocab_size) {
	tokenizer->vocab_size = vocab_size;
	tokenizer->bos_id = bos_id;
	tokenizer->eos_id = eos_id;

	// malloc space to hold the scores and the strings
	tokenizer->vocab = (char**)malloc(vocab_size * sizeof(char*));
	tokenizer->sorted_vocab = (struct TokenIndex*)malloc(vocab_size * sizeof(struct TokenIndex));

	// TODO: validate tokens are null terminated
	for (int i = 0; i < vocab_size; ++i) {
		tokenizer->vocab[i] = tokens;
		tokenizer->sorted_vocab[i].str = tokens;
		tokenizer->sorted_vocab[i].id = i;

		assert(strlen(tokens) <= MAX_TOKEN_LENGTH);
		tokens += strlen(tokens) + 1;
	}

	tokenizer->vocab_scores = scores;

	qsort(tokenizer->sorted_vocab, vocab_size, sizeof(struct TokenIndex), compare_tokens);

	tokenizer->byte_fallbacks = str_lookup("<0x00>", tokenizer->sorted_vocab, vocab_size);

	if (tokenizer->byte_fallbacks >= 0) {
		for (int i = 0; i < 256; i++) {
			tokenizer->byte_pieces[i][0] = (char)i;
			tokenizer->byte_pieces[i][1] = '\0';
		}
	}
}

void tokenizer_free(struct Tokenizer* tokenizer) {
	free(tokenizer->vocab);
	free(tokenizer->sorted_vocab);
}

int tokenizer_bound(int bytes) {
	return bytes + 3; // +3 for prefix space, ?BOS, ?EOS
}

char* tokenizer_decode(struct Tokenizer* tokenizer, int prev_token, int token) {
	char* piece = tokenizer->vocab[token];
	// following BOS token, sentencepiece decoder strips any leading whitespace (see PR #89)
	if (prev_token == tokenizer->bos_id && piece[0] == ' ') {
		piece++;
	}
	// return byte piece for byte fallback tokens (<0x00>, <0x01>, etc.)
	if (tokenizer->byte_fallbacks >= 0 && (unsigned)(token - tokenizer->byte_fallbacks) < 256) {
		piece = tokenizer->byte_pieces[token - tokenizer->byte_fallbacks];
	}
	return piece;
}

int tokenizer_encode(struct Tokenizer* tokenizer, char* text, unsigned flags, int* tokens) {
	// encode the string text (input) into an upper-bound preallocated tokens[] array
	assert(text);
	int n_tokens = 0;

	// create a temporary buffer that will store merge candidates of always two consecutive tokens
	// *2 for concat, +1 for null terminator
	char str_buffer[MAX_TOKEN_LENGTH * 2 + 1];
	size_t str_len = 0;

	// add optional BOS token, if desired
	if (flags & TF_ENCODE_BOS)
		tokens[n_tokens++] = tokenizer->bos_id;

	// add_dummy_prefix is true by default
	// so prepend a dummy prefix token to the input string, but only if text != ""
	// TODO: pretty sure this isn't correct in the general case but I don't have the
	// energy to read more of the sentencepiece code to figure out what it's doing
	if (text[0] != '\0') {
		int dummy_prefix = str_lookup(" ", tokenizer->sorted_vocab, tokenizer->vocab_size);
		tokens[n_tokens++] = dummy_prefix;
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
		int id = str_lookup(str_buffer, tokenizer->sorted_vocab, tokenizer->vocab_size);

		if (id != -1) {
			// we found this codepoint in vocab, add it as a token
			tokens[n_tokens++] = id;
		} else if (tokenizer->byte_fallbacks >= 0) {
			// byte_fallback encoding: just encode each byte as a token
			for (int i = 0; i < str_len; i++) {
				tokens[n_tokens++] = (unsigned char)str_buffer[i] + tokenizer->byte_fallbacks;
			}
		}
		str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
	}

	// merge the best consecutive pair each iteration, according the scores in vocab_scores
	while (1) {
		float best_score = -1e10;
		int best_id = -1;
		int best_idx = -1;

		for (int i = 0; i < n_tokens - 1; i++) {
			// check if we can merge the pair (tokens[i], tokens[i+1])
			strcpy(str_buffer, tokenizer->vocab[tokens[i]]);
			strcat(str_buffer, tokenizer->vocab[tokens[i + 1]]);
			int id = str_lookup(str_buffer, tokenizer->sorted_vocab, tokenizer->vocab_size);
			if (id != -1 && tokenizer->vocab_scores[id] > best_score) {
				// this merge pair exists in vocab! record its score and position
				best_score = tokenizer->vocab_scores[id];
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
		for (int i = best_idx + 1; i < n_tokens - 1; i++) {
			tokens[i] = tokens[i + 1];
		}
		n_tokens--; // token length decreased
	}

	// add optional EOS token, if desired
	if (flags & TF_ENCODE_EOS)
		tokens[n_tokens++] = tokenizer->eos_id;

	assert(n_tokens <= tokenizer_bound(strlen(text)));
	return n_tokens;
}
