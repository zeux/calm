#include <metal_stdlib>
#include <metal_compute>
#include <metal_simdgroup>
using namespace metal;

template <typename T> struct ActT { typedef float type; };
template <> struct ActT<uint8_t> { typedef half type; };
template <> struct ActT<uint32_t> { typedef half type; };
template <typename T> using Act = typename ActT<T>::type;

static constant int warpSize = 32;

inline float blockreduce_sum(threadgroup float* vs, float val, uint id) {
	val = simd_sum(val);

	vs[id / warpSize] = val;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	return simd_sum(vs[id % warpSize]);
}

inline float blockreduce_max(threadgroup float* vs, float val, uint id) {
	val = simd_max(val);

	vs[id / warpSize] = val;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	return simd_max(vs[id % warpSize]);
}

inline half gf4_ff(uint32_t v, int k) {
	half s = as_type<half>(uint16_t(v << 8)) * half(-0.25f); // we expect compiler to reuse this across multiple calls
	return half((int((v >> (8 + k * 3)) & 7) ^ 4) - 4) * s;
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

template <typename AT>
inline float matmul_warppar(device AT* x, device uint8_t* w, int i, int n, uint id) {
	typedef __attribute__((__ext_vector_type__(4))) AT AT4;

	int lane = id % warpSize;
	float val = 0.0f;
	for (int j = lane * 4; j < n; j += warpSize * 4) {
		uint32_t ww = *(device uint32_t*)&w[i * n + j];
		uint32_t wwh = ww >> 16;
		AT4 xx = *(device AT4*)&x[j];
		val += as_type<half>(uint16_t(ww << 8)) * xx.x;
		val += as_type<half>(uint16_t(ww & 0xff00)) * xx.y;
		val += as_type<half>(uint16_t(wwh << 8)) * xx.z;
		val += as_type<half>(uint16_t(wwh & 0xff00)) * xx.w;
	}
	return simd_sum(val);
}

template <typename AT>
inline float matmul_warppar(device AT* x, device uint32_t* w, int i, int n, uint id) {
	typedef __attribute__((__ext_vector_type__(4))) AT AT4;

	int lane = id % warpSize;
	float val = 0.0f;
	for (int j = lane * 8; j < n; j += warpSize * 8) {
		uint32_t wg = w[i * n / 8 + j / 8];
		AT4 xx0 = *(device AT4*)&x[j];
		AT4 xx1 = *(device AT4*)&x[j + 4];

		int wgi = ((wg & 0xfff00000) | ((wg >> 4) & 0xfff0));

		float us = as_type<half>(uint16_t(wg << 8));
		float s = us * -0.25f * exp2(-13.f);

		float acc = 0;
		for (int k = 0; k < 4; ++k) {
			int wgk = wgi << (9 - k * 3);
			if (k != 0) wgk &= 0xe000e000;
			short2 wgkp = as_type<short2>(wgk);
			acc += float(wgkp.x) * xx0[k];
			acc += float(wgkp.y) * xx1[k];
		}
		val += acc * s;
	}
	return simd_sum(val);
}

kernel void prepare_gf4(constant int& n [[buffer(0)]], device uint32_t* data [[buffer(1)]], uint id [[thread_position_in_grid]]) {
	uint32_t wg = data[id];
	wg ^= 0x92492400;
	data[id] = wg;
}

inline float gelu(float x) {
	return 0.5f * x * (1.0f + precise::tanh(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + exp(-x));
}

inline float embed(device half* w, int i) {
	return w[i];
}

inline float embed(device uint8_t* w, int i) {
	return as_type<half>(uint16_t(w[i] << 8));
}

inline float embed(device uint32_t* w, int i) {
	return gf4_ff(w[i / 8], i % 8);
}

template <typename T>
kernel void kernel_embed(constant int& token_offset [[buffer(0)]], device float* o [[buffer(1)]], device T* weight [[buffer(2)]], uint id [[thread_position_in_grid]]) {
	o[id] = embed(weight, id + token_offset);
}

template [[host_name("embed_half")]] kernel void kernel_embed<half>(constant int&, device float*, device half*, uint);
template [[host_name("embed_fp8")]] kernel void kernel_embed<uint8_t>(constant int&, device float*, device uint8_t*, uint);
template [[host_name("embed_gf4")]] kernel void kernel_embed<uint32_t>(constant int&, device float*, device uint32_t*, uint);

struct SinkArgs {
	int kv_dim;
	int head_dim;
	int rotary_dim;

	int kv_sink;
	int seq_len;

	float theta_log2;
};

template <typename KVT>
kernel void kernel_rotate_sink(constant SinkArgs& args [[buffer(0)]], device KVT* keyc [[buffer(1)]], uint id [[thread_position_in_grid]]) {
	int i = (id * 2) % (args.kv_sink * args.kv_dim);
	int l = id / (args.kv_sink * args.kv_dim / 2);

	int j_head = i % args.head_dim;
	float freq = j_head >= args.rotary_dim ? 0.f : exp2(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);

	// rotate sink tokens forward to keep pace with non-sink tokens
	float fcr;
	float fci = sincos(freq, fcr);

	size_t loff = (size_t)l * args.seq_len * args.kv_dim;
	device KVT* kb = keyc + loff;

	// note: k layout is transposed / tiled to improve attn_score performance
	int t = i / args.kv_dim;
	int k = i % args.kv_dim;
	int o = t * 16 + args.seq_len * (k / 16) * 16 + (k % 16);

	float v0 = float(kb[o + 0]);
	float v1 = float(kb[o + 1]);

	float r0 = v0 * fcr - v1 * fci;
	float r1 = v0 * fci + v1 * fcr;

	kb[o + 0] = KVT(r0);
	kb[o + 1] = KVT(r1);
}

template [[host_name("rotate_sink_half")]] kernel void kernel_rotate_sink<half>(constant SinkArgs&, device half*, uint);

struct NormArgs {
	int size;
	float eps;
	bool ln;
};

template <typename T>
kernel void kernel_rmsnorm(constant NormArgs& args [[buffer(0)]], device T* o [[buffer(1)]], device float* x [[buffer(2)]], device float* weight [[buffer(3)]], uint id [[thread_position_in_grid]]) {
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
		mean = blockreduce_sum(vs, sum, i) / size;
	}

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i; j < size; j += blockSize) {
		float v = x[j] - mean;
		ss += v * v;
	}

	// sum across threads in block
	ss = blockreduce_sum(vs, ss, i);

	float scale = rsqrt(ss / size + args.eps);

	for (int j = i; j < size; j += blockSize) {
		o[j] = (x[j] - mean) * weight[j] * scale;
	}
}

template [[host_name("rmsnorm_float")]] kernel void kernel_rmsnorm<float>(constant NormArgs&, device float*, device float*, device float*, uint);
template [[host_name("rmsnorm_half")]] kernel void kernel_rmsnorm<half>(constant NormArgs&, device half*, device float*, device float*, uint);

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
kernel void kernel_qkv(constant QkvArgs& args [[buffer(0)]], device Act<T>* x [[buffer(1)]], device float* qout [[buffer(2)]], device KVT* keyc [[buffer(3)]], device KVT* valc [[buffer(4)]], device T* wq [[buffer(5)]], device T* wk [[buffer(6)]], device T* wv [[buffer(7)]], device float* bqkv [[buffer(8)]], uint id [[thread_position_in_grid]]) {
	int dim = args.dim;
	int q_dim = args.q_dim;
	int kv_dim = args.kv_dim;

	int j = (id / warpSize) * 2;
	device T* w = j < q_dim ? wq : (j < q_dim + kv_dim ? wk : wv);
	int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);

	float v0 = matmul_warppar(x, w, k + 0, dim, id);
	float v1 = matmul_warppar(x, w, k + 1, dim, id);

	v0 += bqkv[j + 0];
	v1 += bqkv[j + 1];

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
template [[host_name("qkv_fp8_half")]] kernel void kernel_qkv<uint8_t, half>(constant QkvArgs&, device half*, device float*, device half*, device half*, device uint8_t*, device uint8_t*, device uint8_t*, device float*, uint);
template [[host_name("qkv_gf4_half")]] kernel void kernel_qkv<uint32_t, half>(constant QkvArgs&, device half*, device float*, device half*, device half*, device uint32_t*, device uint32_t*, device uint32_t*, device float*, uint);

inline float4 attn_load4(device half* p) {
	return float4(*(device half4*)p);
}

template <typename KVT>
inline float attn_score(device KVT* kht, device float* qh, int head_dim, int seq_len, int t, int off) {
	float score = 0.0f;
	for (int j = 0; j < head_dim; j += 16) {
		float4 kk = attn_load4(&kht[j * seq_len + t * 16 + off]);
		float4 qq = *(device float4*)&qh[j + off];
		score += kk.x * qq.x;
		score += kk.y * qq.y;
		score += kk.z * qq.z;
		score += kk.w * qq.w;
	}

	return score;
}

template <typename KVT>
inline float attn_warpdot(device KVT* val, device float* atth, int kv_len, uint id) {
	int kv_len4 = kv_len & ~3;
	int lane = id % warpSize;

	float res = 0.0f;
	float sum = 0.0f;
	for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
		float4 vv = attn_load4(&val[t]);
		float4 aa = *(device float4*)&atth[t];
		res += vv.x * aa.x;
		res += vv.y * aa.y;
		res += vv.z * aa.z;
		res += vv.w * aa.w;
		sum += aa.x + aa.y + aa.z + aa.w;
	}

	if (kv_len4 + lane < kv_len) {
		float a = atth[kv_len4 + lane];
		res += a * float(val[kv_len4 + lane]);
		sum += a;
	}

	res = simd_sum(res);
	sum = simd_sum(sum);

	return res / sum;
}

struct AttnArgs {
	int seq_len;
	int kv_len;
	int head_dim;
	int kv_mul;
	int n_heads;

	size_t loff;
};

template <typename KVT>
kernel void kernel_attn_score(constant AttnArgs& args [[buffer(0)]], device float* att [[buffer(1)]], device float* q [[buffer(2)]], device KVT* keyc [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;

	int h = j % args.n_heads;
	int kvh = h / args.kv_mul;
	int t = (j / args.n_heads) * 8 + (id % warpSize) / 4;

	if (t < args.kv_len) {
		device float* qh = q + h * args.head_dim;
		device KVT* kh = keyc + args.loff + kvh * args.head_dim * args.seq_len;
		device float* atth = att + h * args.seq_len;

		float score = attn_score(kh, qh, args.head_dim, args.seq_len, t, 4 * (id % 4));

		// reduce score across threads in warp; every 4 threads are processing the same output score
		score += simd_shuffle_xor(score, 2);
		score += simd_shuffle_xor(score, 1);
		score /= sqrt(float(args.head_dim));

		atth[t] = score;
	}
}

template [[host_name("attn_score_half")]] kernel void kernel_attn_score<half>(constant AttnArgs&, device float*, device float*, device half*, uint);

[[host_name("attn_softmax")]] kernel void kernel_attn_softmax(constant AttnArgs& args [[buffer(0)]], device float* att [[buffer(1)]], uint id [[thread_position_in_grid]]) {
	const int blockSize = 1024;
	int h = id / blockSize;
	device float* atth = att + h * args.seq_len;

	int i = id % blockSize;
	int size = args.kv_len;
	device float* x = atth;

	threadgroup float vs[32];

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < size; j += blockSize) {
		max_val = max(max_val, x[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(vs, max_val, i);

	// exp per thread
	for (int j = i; j < size; j += blockSize) {
		x[j] = exp(x[j] - max_val);
	}
}

template <typename KVT, typename AT>
kernel void kernel_attn_mix(constant AttnArgs& args [[buffer(0)]], device AT* qout [[buffer(1)]], device float* att [[buffer(2)]], device KVT* valc [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;

	int h = j / args.head_dim;
	int kvh = h / args.kv_mul;
	int j_head = j % args.head_dim;

	device float* atth = att + h * args.seq_len;
	device KVT* vh = valc + args.loff + kvh * args.head_dim * args.seq_len;
	device KVT* val = vh + j_head * args.seq_len;

	float res = attn_warpdot(val, atth, args.kv_len, id);

	if (id % warpSize == 0) {
		qout[j] = res;
	}
}

template [[host_name("attn_mix_half_float")]] kernel void kernel_attn_mix<half, float>(constant AttnArgs&, device float*, device float*, device half*, uint);
template [[host_name("attn_mix_half_half")]] kernel void kernel_attn_mix<half, half>(constant AttnArgs&, device half*, device float*, device half*, uint);

template <typename T>
kernel void kernel_attn_out(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device Act<T>* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w, j, n, id);

	if (id % warpSize == 0) {
		xout[j] += val;
	}
}

template [[host_name("attn_out_half")]] kernel void kernel_attn_out<half>(constant int&, device float*, device float*, device half*, uint);
template [[host_name("attn_out_fp8")]] kernel void kernel_attn_out<uint8_t>(constant int&, device float*, device half*, device uint8_t*, uint);
template [[host_name("attn_out_gf4")]] kernel void kernel_attn_out<uint32_t>(constant int&, device float*, device half*, device uint32_t*, uint);

inline void moe_gate_warp(device float* moebuf, threadgroup float* weights, int experts, int active, uint id) {
	int i = id;

	// (unscaled) softmax across experts
	float w = (i < experts) ? weights[i] : -FLT_MAX;
	float max_val = simd_max(w);
	w = exp(w - max_val);

	// weight in top 24 bits, index in bottom 8
	int wi = (as_type<int>(w) & 0xffffff00) | i;

	// top k within warp
	float sumw = 0.f;
	int acti = -1;

	for (int k = 0; k < active; ++k) {
		int maxi = simd_max(wi);

		sumw += as_type<float>(maxi);

		// keeps top weight in thread k, clears weight for thread with max thread to avoid re-selection
		acti = (i == k) ? maxi : acti;
		wi = (wi == maxi) ? 0 : wi;
	}

	// write normalized weights
	if (i < active) {
		assert(acti >= 0);

		moebuf[i * 2 + 0] = as_type<float>(acti) / sumw;
		*(device int*)&moebuf[i * 2 + 1] = acti & 0xff;
	}
}

template <typename T>
kernel void kernel_moe_gate(constant int* args [[buffer(0)]], device float* moebuf [[buffer(1)]], device Act<T>* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int n = args[0];
	int experts = args[1];
	int active = args[2];

	int j = id / warpSize;
	float v = matmul_warppar(x, w, j, n, id);

	threadgroup float ws[32];
	ws[j] = v;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	if (id < warpSize) {
		moe_gate_warp(moebuf, ws, experts, active, id);
	}
}

template [[host_name("moe_gate_half")]] kernel void kernel_moe_gate<half>(constant int*, device float*, device float*, device half*, uint);
template [[host_name("moe_gate_fp8")]] kernel void kernel_moe_gate<uint8_t>(constant int*, device float*, device half*, device uint8_t*, uint);
template [[host_name("moe_gate_gf4")]] kernel void kernel_moe_gate<uint32_t>(constant int*, device float*, device half*, device uint32_t*, uint);

template <typename T, bool act_gelu>
kernel void kernel_ffn13(constant int* args [[buffer(0)]], device Act<T>* xout [[buffer(1)]], device Act<T>* x [[buffer(2)]], device float* moebuf [[buffer(3)]], device T* w1 [[buffer(4)]], device T* w3 [[buffer(5)]], uint2 id [[thread_position_in_grid]]) {
	int n = args[0];
	int d = args[1];

	int j = id.x / warpSize;
	int e = id.y;
	int je = j + ((device int*)moebuf)[e * 2 + 1] * d;

	float v1 = matmul_warppar(x, w1, je, n, id.x);
	float v3 = matmul_warppar(x, w3, je, n, id.x);

	if (id.x % warpSize == 0) {
		xout[j + e * d] = (act_gelu ? gelu(v1) : silu(v1)) * v3;
	}
}

template [[host_name("ffn13_silu_half")]] kernel void kernel_ffn13<half, false>(constant int*, device float*, device float*, device float*, device half*, device half*, uint2);
template [[host_name("ffn13_silu_fp8")]] kernel void kernel_ffn13<uint8_t, false>(constant int*, device half*, device half*, device float*, device uint8_t*, device uint8_t*, uint2);
template [[host_name("ffn13_silu_gf4")]] kernel void kernel_ffn13<uint32_t, false>(constant int*, device half*, device half*, device float*, device uint32_t*, device uint32_t*, uint2);

template [[host_name("ffn13_gelu_half")]] kernel void kernel_ffn13<half, true>(constant int*, device float*, device float*, device float*, device half*, device half*, uint2);
template [[host_name("ffn13_gelu_fp8")]] kernel void kernel_ffn13<uint8_t, true>(constant int*, device half*, device half*, device float*, device uint8_t*, device uint8_t*, uint2);
template [[host_name("ffn13_gelu_gf4")]] kernel void kernel_ffn13<uint32_t, true>(constant int*, device half*, device half*, device float*, device uint32_t*, device uint32_t*, uint2);

template <typename T>
kernel void kernel_ffn2(constant int* args [[buffer(0)]], device float* xout [[buffer(1)]], device Act<T>* x [[buffer(2)]], device float* moebuf [[buffer(3)]], device T* w2 [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
	int n = args[0];
	int d = args[1];

	int j = id.x / warpSize;
	int e = id.y;
	int je = j + ((device int*)moebuf)[e * 2 + 1] * d;

	float val = matmul_warppar(x + e * n, w2, je, n, id.x);

	if (id.x % warpSize == 0) {
		atomic_fetch_add_explicit((volatile device atomic_float*)&xout[j], val * moebuf[e * 2 + 0], memory_order_relaxed);
	}
}

template [[host_name("ffn2_half")]] kernel void kernel_ffn2<half>(constant int*, device float*, device float*, device float*, device half*, uint2);
template [[host_name("ffn2_fp8")]] kernel void kernel_ffn2<uint8_t>(constant int*, device float*, device half*, device float*, device uint8_t*, uint2);
template [[host_name("ffn2_gf4")]] kernel void kernel_ffn2<uint32_t>(constant int*, device float*, device half*, device float*, device uint32_t*, uint2);

template <typename T>
kernel void kernel_output(constant int& n [[buffer(0)]], device float* xout [[buffer(1)]], device Act<T>* x [[buffer(2)]], device T* w [[buffer(3)]], uint id [[thread_position_in_grid]]) {
	int j = id / warpSize;
	float val = matmul_warppar(x, w, j, n, id);

	if (id % warpSize == 0) {
		xout[j] = val;
	}
}

template [[host_name("output_half")]] kernel void kernel_output<half>(constant int&, device float*, device float*, device half*, uint);
template [[host_name("output_fp8")]] kernel void kernel_output<uint8_t>(constant int&, device float*, device half*, device uint8_t*, uint);
template [[host_name("output_gf4")]] kernel void kernel_output<uint32_t>(constant int&, device float*, device half*, device uint32_t*, uint);
