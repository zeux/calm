#include <metal_stdlib>
#include <metal_compute>
#include <metal_simdgroup>
using namespace metal;

static constant int warpSize = 32;

inline float blockreduce_sum(threadgroup float* vs, float val, uint id) {
	val = simd_sum(val);

	vs[id / warpSize] = val;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	return simd_sum(vs[id % warpSize]);
}

inline float matmul_warppar(device float* x, device half* w, int i, int n, uint id) {
	int lane = id % warpSize;
	float val = 0.0f;
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		float2 ww = float2(*(device half2*)&w[i * n + j]);
		float2 xx = *(device float2*)&x[j];
		val += ww.x * xx.x;
		val += ww.y * xx.y;
	}
	return simd_sum(val);
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + exp(-x));
}

template <typename T>
kernel void kernel_embed(constant int& token_offset [[buffer(0)]], device float* o [[buffer(1)]], device T* weight [[buffer(2)]], uint id [[thread_position_in_grid]]) {
	o[id] = weight[id + token_offset];
}

template [[host_name("embed_half")]] kernel void kernel_embed<half>(constant int&, device float*, device half*, uint);

struct NormArgs {
	int size;
	float eps;
	bool ln;
};

[[host_name("rmsnorm")]] kernel void kernel_rmsnorm(constant NormArgs& args [[buffer(0)]], device float* o [[buffer(1)]], device float* x [[buffer(2)]], device float* weight [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int i = id;
	const int blockSize = 1024;
	int size = args.size;

	threadgroup float vs[32];

	float mean = 0.0f;
	if (args.ln) {
		// calculate sum (per thread)
		float sum = 0.0f;
		for (int j = i; j < size; j += blockSize) {
			sum += x[j];
		}

		// sum across threads in block
		mean = blockreduce_sum(vs, sum, id) / size;
	}

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j] - mean;
		ss += v * v;
	}

	// sum across threads in block
	ss = blockreduce_sum(vs, ss, id);

	float scale = rsqrt(ss / size + args.eps);

	for (int j = i; j < size; j += blockSize) {
		o[j] = (x[j] - mean) * weight[j] * scale;
	}
}

struct QkvArgs {
	int dim;
	int q_dim;
	int kv_dim;
	int head_dim;
	int rotary_dim;

	int pos;
	int kv_pos;
	int seq_len;

	size_t loff;

	float qkv_clip;
	float theta_log2;
};

template <typename T, typename KVT>
kernel void kernel_qkv(constant QkvArgs& args [[buffer(0)]], device float* x [[buffer(1)]], device float* qout [[buffer(2)]], device KVT* keyc [[buffer(3)]], device KVT* valc [[buffer(4)]], device T* wq [[buffer(5)]], device T* wk [[buffer(6)]], device T* wv [[buffer(7)]], device float* bqkv [[buffer(8)]], uint id [[thread_position_in_grid]]) {
	int dim = args.dim;
	int q_dim = args.q_dim;
	int kv_dim = args.kv_dim;

	int j = id / warpSize;
	device T* w = j < q_dim ? wq : (j < q_dim + kv_dim ? wk : wv);
	int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);

	float v0 = matmul_warppar(x, w, k + 0, dim, id);
	float v1 = matmul_warppar(x, w, k + 1, dim, id);

	if (bqkv) {
		v0 += bqkv[j + 0];
		v1 += bqkv[j + 1];
	}

	v0 = min(max(v0, -args.qkv_clip), args.qkv_clip);
	v1 = min(max(v1, -args.qkv_clip), args.qkv_clip);

	if (id % warpSize == 0) {
		int j_head = j % args.head_dim;
		float freq = j_head >= args.rotary_dim ? 0.f : exp2(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);
		float fcr;
		float fci = sincos(args.pos * freq, fcr);

		if (j < q_dim) {
			qout[k + 0] = v0 * fcr - v1 * fci;
			qout[k + 1] = v0 * fci + v1 * fcr;
		} else if (j < q_dim + kv_dim) {
			// note: k layout is transposed / tiled to improve attn_score performance
			int off = args.kv_pos * 16 + args.seq_len * (k / 16) * 16 + (k % 16);
			keyc[args.loff + off + 0] = KVT(v0 * fcr - v1 * fci);
			keyc[args.loff + off + 1] = KVT(v0 * fci + v1 * fcr);
		} else {
			// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
			valc[args.loff + args.kv_pos + args.seq_len * (k + 0)] = KVT(v0);
			valc[args.loff + args.kv_pos + args.seq_len * (k + 1)] = KVT(v1);
		}
	}
}

template [[host_name("qkv_half_half")]] kernel void kernel_qkv<half, half>(constant QkvArgs&, device float*, device float*, device half*, device half*, device half*, device half*, device half*, device float*, uint);

template <typename T>
kernel void kernel_attn_out(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w, j, n, id);

	if (id % warpSize == 0) {
		xout[j] += val;
	}
}

template [[host_name("attn_out_half")]] kernel void kernel_attn_out<half>(constant int&, device float*, device float*, device half*, uint);

template <typename T, bool act_gelu>
kernel void kernel_ffn13(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w1 [[buffer(3)]], device T* w3 [[buffer(4)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float v1 = matmul_warppar(x, w1, j, n, id);
	float v3 = matmul_warppar(x, w3, j, n, id);

	if (id % warpSize == 0) {
		xout[j] = (act_gelu ? gelu(v1) : silu(v1)) * v3;
	}
}

template [[host_name("ffn13_silu_half")]] kernel void kernel_ffn13<half, false>(constant int&, device float*, device float*, device half*, device half*, uint);
template [[host_name("ffn13_gelu_half")]] kernel void kernel_ffn13<half, true>(constant int&, device float*, device float*, device half*, device half*, uint);

template <typename T>
kernel void kernel_ffn2(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w2 [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w2, j, n, id);

	if (id % warpSize == 0) {
		xout[j] += val;
	}
}

template [[host_name("ffn2_half")]] kernel void kernel_ffn2<half>(constant int&, device float*, device float*, device half*, uint);

template <typename T>
kernel void kernel_output(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device float* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w, j, n, id);

	if (id % warpSize == 0) {
		xout[j] = val;
	}
}

template [[host_name("output_half")]] kernel void kernel_output<half>(constant int&, device float*, device float*, device half*, uint);
