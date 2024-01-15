#pragma once

#include <stddef.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#define MAX_LAYERS 128

// Can switch between float and _Float16
#ifdef __CUDACC__
typedef half kvtype_t;
#elif defined(__FLT16_MANT_DIG__)
typedef _Float16 kvtype_t;
#else
// We can't use _Float16 on CPU but we can still run CUDA
typedef short kvtype_t;
#endif

// How many attention sinks to use for rolling buffer
#define KV_SINKS 2

enum Arch {
	LlamaLike,
	Phi
};

struct Config {
	enum Arch arch;   // model architecture
	int dim;          // transformer dimension
	int hidden_dim;   // for ffn layers
	int n_layers;     // number of layers
	int n_heads;      // number of query heads
	int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
	int vocab_size;   // vocabulary size, usually 256 (byte-level)
	int seq_len;      // max sequence length
	float rope_theta; // RoPE theta
	int rotary_dim;   // RoPE rotary dimension (elements after that don't get rotated)
};

struct Weights {
	int dbits; // 8 for fp8, 16 for fp16; determines type of void* below

	// token embedding table
	void* token_embedding_table; // (vocab_size, dim)
	// weights for layernorm (phi)
	float* ln_weight[MAX_LAYERS]; // (dim,)
	float* ln_bias[MAX_LAYERS];   // (dim,)
	// weights for rmsnorms
	float* rms_att_weight[MAX_LAYERS]; // (dim) rmsnorm weights
	float* rms_ffn_weight[MAX_LAYERS]; // (dim)
	// weights for matmuls. note dim == n_heads * head_size
	void* wq[MAX_LAYERS]; // (dim, n_heads * head_size)
	void* wk[MAX_LAYERS]; // (dim, n_kv_heads * head_size)
	void* wv[MAX_LAYERS]; // (dim, n_kv_heads * head_size)
	void* wo[MAX_LAYERS]; // (n_heads * head_size, dim)
	// weights for ffn (w3 is absent for phi)
	void* w1[MAX_LAYERS]; // (hidden_dim, dim)
	void* w2[MAX_LAYERS]; // (dim, hidden_dim)
	void* w3[MAX_LAYERS]; // (hidden_dim, dim)
	// final layernorm (phi)
	float* ln_final_weight; // (dim,)
	float* ln_final_bias;   // (dim,)
	// final rmsnorm
	float* rms_final_weight; // (dim,)
	// classifier weights for the logits, on the last layer
	void* wcls;
	// biases for all of the above (phi)
	float* bq[MAX_LAYERS]; // (dim)
	float* bk[MAX_LAYERS]; // (dim)
	float* bv[MAX_LAYERS]; // (dim)
	float* bo[MAX_LAYERS]; // (dim)
	float* b1[MAX_LAYERS]; // (hidden_dim)
	float* b2[MAX_LAYERS]; // (dim)
	float* bcls;
};

struct RunState {
	// current wave of activations
	float* x;      // activation at current time stamp (dim,)
	float* xb;     // same, but inside a residual branch (dim,)
	float* xb2;    // an additional buffer just for convenience (dim,)
	float* xa;     // buffer for parallel activation accumulation (dim,)
	float* hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
	float* hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
	float* q;      // query (dim,)
	float* k;      // key (dim,)
	float* v;      // value (dim,)
	float* att;    // buffer for scores/attention values (n_heads, seq_len)
	float* logits; // output logits
	// kv cache
	kvtype_t* key_cache;   // (layer, seq_len, dim)
	kvtype_t* value_cache; // (layer, seq_len, dim)
};

struct Transformer {
	struct Config config;   // the hyperparameters of the architecture (the blueprint)
	struct Weights weights; // the weights of the model
	struct RunState state;  // buffers for the "wave" of activations in the forward pass
	size_t n_params, n_bytes;
	float* (*forward)(struct Transformer* transformer, int token, int pos, unsigned flags);
};

enum ForwardFlags {
	FF_UPDATE_KV_ONLY = 1 << 0, // only update kv cache and don't output logits
};
