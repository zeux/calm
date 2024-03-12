#include "tokenizer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKEN_LENGTH 512

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

struct Merge {
	int lpos, lid;
	int rpos, rid;
	int resid;
	float score;
};

static void heap_swap(struct Merge* heap, int i, int j) {
	struct Merge tmp = heap[i];
	heap[i] = heap[j];
	heap[j] = tmp;
}

static void heap_insert(struct Merge* heap, int n_heap, struct Merge merge) {
	// insert a new element at the end (breaks heap invariant)
	heap[n_heap] = merge;
	n_heap++;

	// bubble up the new element to its correct position
	int i = n_heap - 1;
	while (i > 0 && heap[i].score > heap[(i - 1) / 2].score) {
		heap_swap(heap, i, (i - 1) / 2);
		i = (i - 1) / 2;
	}
}

static void heap_poptop(struct Merge* heap, int n_heap) {
	// move the last element to the top (breaks heap invariant)
	n_heap--;
	heap[0] = heap[n_heap];

	// bubble down the new top element to its correct position
	int i = 0;
	while (i * 2 + 1 < n_heap) {
		// find the largest child
		int j = i * 2 + 1;
		if (j + 1 < n_heap && heap[j + 1].score > heap[j].score) {
			j++;
		}
		// if the largest child is smaller than the parent, we're done
		if (heap[j].score <= heap[i].score) {
			break;
		}
		// otherwise, swap the parent and child
		heap_swap(heap, i, j);
		i = j;
	}
}

static int merge_tokens_tryadd(struct Tokenizer* tokenizer, struct Merge* heap, int n_heap, int lpos, int lid, int rpos, int rid) {
	char str_buffer[MAX_TOKEN_LENGTH * 2 + 1];
	strcpy(str_buffer, tokenizer->vocab[lid]);
	strcat(str_buffer, tokenizer->vocab[rid]);
	int id = str_lookup(str_buffer, tokenizer->sorted_vocab, tokenizer->vocab_size);
	if (id != -1) {
		struct Merge merge = {lpos, lid, rpos, rid, id, tokenizer->vocab_scores[id]};
		heap_insert(heap, n_heap++, merge);
	}
	return n_heap;
}

static int merge_tokens(struct Tokenizer* tokenizer, int* tokens, int n_tokens) {
	// create heap for all token merge pairs
	struct Merge* heap = malloc(2 * n_tokens * sizeof(struct Merge));
	int n_heap = 0;

	// insert all initial pairs
	for (int i = 0; i < n_tokens - 1; i++) {
		n_heap = merge_tokens_tryadd(tokenizer, heap, n_heap, i, tokens[i], i + 1, tokens[i + 1]);
	}

	// merge all pairs
	while (n_heap > 0) {
		struct Merge merge = heap[0];
		heap_poptop(heap, n_heap--);

		if (tokens[merge.lpos] != merge.lid || tokens[merge.rpos] != merge.rid) {
			continue; // this pair was already merged, skip it
		}

		// merge
		tokens[merge.lpos] = merge.resid;
		tokens[merge.rpos] = -1;

		// we might have new pairs to merge
		for (int i = merge.lpos - 1; i >= 0; i--) {
			if (tokens[i] != -1) {
				n_heap = merge_tokens_tryadd(tokenizer, heap, n_heap, i, tokens[i], merge.lpos, merge.resid);
				break;
			}
		}

		for (int i = merge.rpos + 1; i < n_tokens; i++) {
			if (tokens[i] != -1) {
				n_heap = merge_tokens_tryadd(tokenizer, heap, n_heap, merge.lpos, merge.resid, i, tokens[i]);
				break;
			}
		}
	}

	free(heap);

	// compact tokens
	int nm_tokens = 0;
	for (int i = 0; i < n_tokens; i++) {
		if (tokens[i] != -1) {
			tokens[nm_tokens++] = tokens[i];
		}
	}

	return nm_tokens;
}

int tokenizer_encode(struct Tokenizer* tokenizer, char* text, unsigned flags, int* tokens) {
	int n_tokens = 0;

	// add optional BOS token, if desired
	if ((flags & TF_ENCODE_BOS) && tokenizer->bos_id >= 0) {
		tokens[n_tokens++] = tokenizer->bos_id;
	}

	// assuming add_dummy_prefix is true, prepend a dummy prefix token to the input string, but only if text != ""
	if (text[0] != '\0' && tokenizer->bos_id >= 0) {
		int dummy_prefix = str_lookup(" ", tokenizer->sorted_vocab, tokenizer->vocab_size);
		tokens[n_tokens++] = dummy_prefix;
	}

	// process the raw (UTF-8) byte sequence of the input string
	for (char* c = text; *c != '\0';) {
		char codepoint[5] = {};

		codepoint[0] = *c++;

		if (codepoint[0] == '<' && *c == '|') {
			// special token, skip until '|>'
			char* e = c;
			while (*e && !(e[0] == '|' && e[1] == '>')) {
				e++;
			}
			if (e[0] == '|' && e[1] == '>' && e - c + 3 <= MAX_TOKEN_LENGTH) {
				// we found the end of the special token, try to encode it as is
				char special[MAX_TOKEN_LENGTH + 1];
				memcpy(special, c - 1, e - c + 3);
				special[e - c + 3] = '\0';

				int sid = str_lookup(special, tokenizer->sorted_vocab, tokenizer->vocab_size);
				if (sid != -1) {
					// we found special codepoint in vocab, add it as a token
					tokens[n_tokens++] = sid;
					c = e + 2;
					continue;
				}
			}
		}

		// this byte is a leading byte (11...), so it's a multi-byte UTF8 codepoint
		if ((codepoint[0] & 0xC0) == 0xC0) {
			for (int i = 1; i < 4 && (*c & 0xC0) == 0x80; ++i) {
				codepoint[i] = *c++;
			}
		}

		int id = str_lookup(codepoint, tokenizer->sorted_vocab, tokenizer->vocab_size);

		if (id != -1) {
			// we found this codepoint in vocab, add it as a token
			tokens[n_tokens++] = id;
		} else if (tokenizer->byte_fallbacks >= 0) {
			// byte_fallback encoding: just encode each byte as a token
			for (char* fb = codepoint; *fb != '\0'; ++fb) {
				tokens[n_tokens++] = (unsigned char)*fb + tokenizer->byte_fallbacks;
			}
		}
	}

	// optimized heap-based merge
	n_tokens = merge_tokens(tokenizer, tokens, n_tokens);

	// add optional EOS token, if desired
	if (flags & TF_ENCODE_EOS) {
		tokens[n_tokens++] = tokenizer->eos_id;
	}

	assert(n_tokens <= tokenizer_bound(strlen(text)));
	return n_tokens;
}
