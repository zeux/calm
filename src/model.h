#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#define MAX_LAYERS 128

// Can switch between float and _Float16 (model rebuild required)
#ifdef __CUDACC__
typedef half dtype_t;
typedef half kvtype_t;
#elif defined(__FLT16_MANT_DIG__)
typedef _Float16 dtype_t;
typedef _Float16 kvtype_t;
#else
// We can't use _Float16 on CPU but we can still run CUDA
typedef short dtype_t;
typedef short kvtype_t;
#endif

struct Config {
	int dim;          // transformer dimension
	int hidden_dim;   // for ffn layers
	int n_layers;     // number of layers
	int n_heads;      // number of query heads
	int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
	int vocab_size;   // vocabulary size, usually 256 (byte-level)
	int seq_len;      // max sequence length
	float rope_theta; // RoPE theta
};

struct Weights {
	// token embedding table
	dtype_t* token_embedding_table; // (vocab_size, dim)
	// weights for rmsnorms
	dtype_t* rms_att_weight[MAX_LAYERS]; // (dim) rmsnorm weights
	dtype_t* rms_ffn_weight[MAX_LAYERS]; // (dim)
	// weights for matmuls. note dim == n_heads * head_size
	dtype_t* wq[MAX_LAYERS]; // (dim, n_heads * head_size)
	dtype_t* wk[MAX_LAYERS]; // (dim, n_kv_heads * head_size)
	dtype_t* wv[MAX_LAYERS]; // (dim, n_kv_heads * head_size)
	dtype_t* wo[MAX_LAYERS]; // (n_heads * head_size, dim)
	// weights for ffn
	dtype_t* w1[MAX_LAYERS]; // (hidden_dim, dim)
	dtype_t* w2[MAX_LAYERS]; // (dim, hidden_dim)
	dtype_t* w3[MAX_LAYERS]; // (hidden_dim, dim)
	// final rmsnorm
	dtype_t* rms_final_weight; // (dim,)
	// (optional) classifier weights for the logits, on the last layer
	dtype_t* wcls;
};

struct RunState {
	// current wave of activations
	float* x;      // activation at current time stamp (dim,)
	float* xb;     // same, but inside a residual branch (dim,)
	float* xb2;    // an additional buffer just for convenience (dim,)
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
	float* (*forward)(struct Transformer* transformer, int token, int pos);
};
