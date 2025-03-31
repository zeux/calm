# Produce a safetensors model file out of multiple inputs
# python convert.py model.safetensors --config config.json --models file1.bin file2.bin ...

import argparse
import base64
import json
import os.path
import safetensors
import safetensors.torch
import torch
# optionally imports sentencepiece below when converting models without HF tokenizer.json

argp = argparse.ArgumentParser()
argp.add_argument("output", type=str)
argp.add_argument("input", type=str, nargs="?")
argp.add_argument("--config", type=str)
argp.add_argument("--tokenizer", type=str)
argp.add_argument("--models", type=str, nargs="+")
argp.add_argument("--dtype", type=str, default="fp8", choices=["fp16", "fp8", "gf4"])
args = argp.parse_args()

if args.input is not None:
    # assume input is a directory with HuggingFace layout
    if args.config is None:
        args.config = os.path.join(args.input, "config.json")
        if not os.path.exists(args.config):
            argp.error("no config.json found in {}".format(args.input))
    if args.tokenizer is None:
        args.tokenizer = os.path.join(args.input, "tokenizer.json")
        if not os.path.exists(args.tokenizer):
            args.tokenizer = os.path.join(args.input, "tokenizer.model")
        if not os.path.exists(args.tokenizer):
            argp.error("no tokenizer.json or tokenizer.model found in {}".format(args.input))
    if args.models is None:
        files = os.listdir(args.input)
        args.models = [os.path.join(args.input, fn) for fn in files if os.path.splitext(fn)[1] == ".safetensors"]
        if len(args.models) == 0:
            args.models = [os.path.join(args.input, fn) for fn in files if os.path.splitext(fn)[1] == ".bin"]
        if len(args.models) == 0:
            argp.error("no .safetensors or .bin files found in {}".format(args.input))
elif args.config is None or args.models is None:
    argp.error("arguments --config, --tokenizer and --models are required unless argument input is specified")

with open(args.config, "r") as f:
    config = json.load(f)

metadata = {}
tensors = {}

arch = config["architectures"][0]
arch_remap = {"LlamaForCausalLM": "llama", "MistralForCausalLM": "mistral", "MixtralForCausalLM": "mixtral", "Qwen2ForCausalLM": "qwen2", "OLMoForCausalLM": "olmo", "GemmaForCausalLM": "gemma", "MiniCPMForCausalLM": "minicpm", "CohereForCausalLM": "cohere", "InternLM2ForCausalLM": "internlm2", "DbrxForCausalLM": "dbrx", "XverseForCausalLM": "xverse", "Phi3ForCausalLM": "phi3", "OlmoeForCausalLM": "olmoe"}
assert arch in arch_remap, "Unsupported architecture: {}; must be one of: {}".format(arch, list(arch_remap.keys()))
arch = arch_remap[arch]

metadata["arch"] = arch
metadata["dtype"] = args.dtype

if arch in ["llama", "mistral", "mixtral", "qwen2", "gemma", "minicpm", "cohere", "internlm2", "xverse", "phi3", "olmoe"]:
    metadata["dim"] = config["hidden_size"]
    metadata["hidden_dim"] = config["intermediate_size"]
    metadata["head_dim"] = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    metadata["n_layers"] = config["num_hidden_layers"]
    metadata["n_heads"] = config["num_attention_heads"]
    metadata["n_kv_heads"] = config.get("num_key_value_heads", config["num_attention_heads"])
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = 2048 if arch == "phi3" else config["max_position_embeddings"]
    metadata["bos_token_id"] = -1 if arch in ["qwen2", "olmoe"] else config["bos_token_id"]
    metadata["eos_token_id"] = config["eos_token_id"]
    metadata["rope_theta"] = config.get("rope_theta", 10000.0)
    metadata["rotary_dim"] = int(metadata["head_dim"] * config.get("partial_rotary_factor", 1))
    metadata["norm_eps"] = config["layer_norm_eps"] if arch == "cohere" else config["rms_norm_eps"]
    metadata["norm_type"] = "layernorm_par" if arch == "cohere" else "rmsnorm"

    assert config["hidden_act"] in ["gelu", "silu"]
    metadata["act_type"] = config["hidden_act"]

    # moe
    if arch in ["mixtral"]:
        metadata["n_experts"] = config["num_local_experts"]
        metadata["n_experts_active"] = config["num_experts_per_tok"]
    elif arch in ["minicpm"] and "num_experts" in config:
        metadata["n_experts"] = config["num_experts"]
        metadata["n_experts_active"] = config["num_experts_per_tok"]
    elif arch in ["olmoe"]:
        metadata["n_experts"] = config["num_experts"]
        metadata["n_experts_active"] = config["num_experts_per_tok"]
elif arch == "olmo":
    metadata["dim"] = config["d_model"]
    metadata["hidden_dim"] = (config["mlp_hidden_size"] or config["d_model"] * config["mlp_ratio"]) // 2
    metadata["n_layers"] = config["n_layers"]
    metadata["n_heads"] = config["n_heads"]
    metadata["n_kv_heads"] = config["n_heads"]
    metadata["vocab_size"] = config["embedding_size"]
    metadata["max_seq_len"] = config["max_sequence_length"]
    metadata["bos_token_id"] = -1
    metadata["eos_token_id"] = config["eos_token_id"]
    metadata["rope_theta"] = 10000.0
    metadata["rotary_dim"] = config["d_model"] // config["n_heads"]
    metadata["norm_eps"] = 1e-5
    metadata["norm_type"] = "layernorm"

    assert config["activation_type"] == "swiglu"
    metadata["act_type"] = "silu"

    if config.get("clip_qkv", None):
        metadata["qkv_clip"] = config["clip_qkv"]
elif arch == "dbrx":
    metadata["dim"] = config["d_model"]
    metadata["hidden_dim"] = config["ffn_config"]["ffn_hidden_size"]
    metadata["head_dim"] = config["d_model"] // config["n_heads"]
    metadata["n_layers"] = config["n_layers"]
    metadata["n_heads"] = config["n_heads"]
    metadata["n_kv_heads"] = config["attn_config"]["kv_n_heads"]
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = config["max_seq_len"]
    metadata["bos_token_id"] = -1
    metadata["eos_token_id"] = 100257
    metadata["rope_theta"] = config["attn_config"]["rope_theta"]
    metadata["rotary_dim"] = config["d_model"] // config["n_heads"]
    metadata["norm_eps"] = 1e-5
    metadata["norm_type"] = "layernorm"
    metadata["act_type"] = "silu"
    metadata["n_experts"] = config["ffn_config"]["moe_num_experts"]
    metadata["n_experts_active"] = config["ffn_config"]["moe_top_k"]
    metadata["qkv_clip"] = config["attn_config"]["clip_qkv"]

# this is a horrible gpt-2 unicode byte encoder hack from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
# this has poisoned all HF tokenizer configs that use ByteLevel decoder/preprocessor
# as a result we get crazy UTF-8-as-bytes-as-UTF8 in the tokenizer data that we need to convert back
def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# load tokenizer model
tokens = [""] * metadata["vocab_size"]
scores = [0] * metadata["vocab_size"]
tokens_gpt2 = False

ext = os.path.splitext(args.tokenizer)[1]
if ext == ".json":
    with open(args.tokenizer, "r") as f:
        tokenizer = json.load(f)

    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= config["vocab_size"]

    tokens_gpt2 = not tokenizer["model"].get("byte_fallback", False)

    for t, i in vocab.items():
        tokens[i] = t

    for added in tokenizer["added_tokens"]:
        tokens[added["id"]] = added["content"]

    # compute score as negative merge index so that earlier merges get selected first
    for i, m in enumerate(tokenizer["model"]["merges"]):
        t1, t2 = (m[0], m[1]) if isinstance(m, list) else m.split(" ", 2)
        ti = vocab[t1 + t2]
        if scores[ti] == 0:
            scores[ti] = -(1 + i)
elif ext == ".model":
    import sentencepiece
    sp_model = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    assert sp_model.vocab_size() <= config["vocab_size"]
    assert sp_model.bos_id() == config["bos_token_id"]
    assert sp_model.eos_id() == config["eos_token_id"]

    for i in range(sp_model.vocab_size()):
        tokens[i] = sp_model.id_to_piece(i)
        scores[i] = sp_model.get_score(i)
elif ext == ".tiktoken":
    with open(args.tokenizer, "r") as f:
        vocab = f.readlines()
    assert len(vocab) <= config["vocab_size"]

    for i, l in enumerate(vocab):
        t, r = l.rstrip().split(" ")
        t = base64.b64decode(t)
        tokens[i] = t.decode("utf-8", errors="replace").replace("\0", "\7")
        scores[i] = -int(r)
else:
    raise Exception("Unknown tokenizer file extension: {}; expected .json or .model/.tiktoken".format(ext))

# postprocess tokens
gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}

for i, t in enumerate(tokens):
    if tokens_gpt2:
        b = bytes([gpt2_decode.get(c, 0) for c in t])
    else:
        t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')

    b = b.replace(b"\0", b"\7") # replace null bytes with bell characters
    assert b.count(0) == 0 # no null bytes allowed

    tokens[i] = b

# load model files
weights = {}
for fn in args.models:
    ext = os.path.splitext(fn)[1]
    if ext == ".safetensors":
        with safetensors.safe_open(fn, framework="pt") as f:
            for k in f.keys():
                assert(k not in weights)
                weights[k] = f.get_tensor(k)
    elif ext == ".bin":
        pth = torch.load(fn, map_location="cpu", weights_only=True)
        for k in pth.keys():
            assert(k not in weights)
            weights[k] = pth[k]
    else:
        raise Exception("Unknown model file extension: {}; expected .safetensors or .bin".format(ext))

# huggingface permutes WQ and WK, this function reverses it
# see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))
    # wr is the rotary part, wk is the part kept unrotated
    wr = w[:, :rotary_dim]
    wk = w[:, rotary_dim:]
    # switch wr from outputting two rotary_dim/2 chunks to outputting values interleaved
    wr = torch.unflatten(wr, 1, (2, -1))
    wr = wr.transpose(1, 2)
    wr = wr.flatten(1, 2)
    # assemble the heads back
    w = torch.cat([wr, wk], dim=1)
    return torch.flatten(w, 0, 1)

# fp8 support requires torch 2.1, but we support other dtypes on earlier versions
dtype = {"fp16": torch.float16, "fp8": getattr(torch, "float8_e5m2", None), "gf4": torch.uint8}[args.dtype]
assert dtype

# gf4 quantization: 8 values get quantized to 32 bits, 3-bit normalized int per value + shared fp8 scale factor
# int range is asymmetric; we use this fact to encode the max value as -4 to expand the range a little bit
def gf4(t):
    if torch.cuda.is_available():
        t.max() # work around cuda load from mmap using small block size for reading...
        t = t.cuda()
    # groups of 8 values
    gt = t.unflatten(-1, (-1, 8))
    # max (abs) of each group
    _, gmaxi = gt.abs().max(-1)
    gmax = gt.gather(-1, gmaxi.unsqueeze(-1))
    # round gmax to fp8 to make sure we're quantizing to the right range
    gmax = gmax.to(torch.float8_e5m2).to(gmax.dtype)
    # normalize gt; note that gmax may be zero
    gt /= gmax
    torch.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0, out=gt)
    # normalize each group by -max ([-1, 1]) and quantize to [0, 8)
    # note that 8 needs to be clamped to 7 since positive half of the range is shorter
    gtq = (gt.to(torch.float16) * -4 + 4).clamp(0, 7).round().to(torch.int32)
    # assemble the results
    gtq <<= torch.tensor([8 + i * 3 for i in range(8)], dtype=torch.int32, device=gtq.device)
    gtr = gtq.sum(-1, dtype=torch.int32)
    gtr += gmax.squeeze(-1).to(torch.float8_e5m2).view(torch.uint8)
    return gtr.cpu()

# preprocess weights
if arch == "minicpm":
    # apply various scaling factors that other models don't have to tensors
    embed_scale = config["scale_emb"]
    resid_scale = config["scale_depth"] / (config["num_hidden_layers"] ** 0.5)
    final_scale = config["dim_model_base"] / config["hidden_size"]

    weights["model.norm.weight"] *= final_scale / (1.0 if config.get("tie_word_embeddings", None) == False else embed_scale)
    weights["model.embed_tokens.weight"] *= embed_scale

    for l in range(config["num_hidden_layers"]):
        weights[f"model.layers.{l}.self_attn.o_proj.weight"] *= resid_scale

        if "num_experts" in config:
            for e in range(config["num_experts"]):
                weights[f"model.layers.{l}.mlp.experts.{e}.w2.weight"] *= resid_scale
        else:
            weights[f"model.layers.{l}.mlp.down_proj.weight"] *= resid_scale
elif arch == "gemma":
    # gemma's norm weights are stored relative to 1.0
    weights["model.norm.weight"] = weights["model.norm.weight"].float() + 1

    for l in range(config["num_hidden_layers"]):
        weights[f"model.layers.{l}.input_layernorm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float() + 1
        weights[f"model.layers.{l}.post_attention_layernorm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float() + 1

    # apply embedding scale (and counter it since output weights are tied)
    # this improves precision for fp8
    embed_scale = config["hidden_size"] ** 0.5

    weights["model.norm.weight"] *= 1 / embed_scale
    weights["model.embed_tokens.weight"] = weights["model.embed_tokens.weight"].float() * embed_scale
elif arch == "cohere":
    weights["model.norm.weight"] *= config["logit_scale"]

# convert weights
progress = 0
def conv(t):
    global progress
    progress += 1
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return gf4(t) if dtype == torch.uint8 else t.to(dtype)

if arch in ["llama", "mistral", "mixtral", "qwen2", "gemma", "minicpm", "cohere", "xverse", "olmoe"]:
    if arch == "olmoe":
        print("Warning: Olmoe uses QK norm which we do not support")

    tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

        rotary_dim = metadata["rotary_dim"]
        n_heads = config["num_attention_heads"]
        n_kv_heads = config.get("num_key_value_heads", n_heads)

        if arch == "cohere":
            tensors[f"model.layers.{l}.attn.wq.weight"] = conv(weights[f"model.layers.{l}.self_attn.q_proj.weight"])
            tensors[f"model.layers.{l}.attn.wk.weight"] = conv(weights[f"model.layers.{l}.self_attn.k_proj.weight"])
        else:
            tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], n_heads, rotary_dim))
            tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], n_kv_heads, rotary_dim))

        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

        if arch in ["qwen2"]:
            tensors[f"model.layers.{l}.attn.wqkv.bias"] = torch.cat([
                permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.bias"], n_heads, rotary_dim).float(),
                permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.bias"], n_kv_heads, rotary_dim).float(),
                weights[f"model.layers.{l}.self_attn.v_proj.bias"].float()
            ])

        if arch != "cohere":
            tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

        if arch in ["mixtral"]:
            tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"model.layers.{l}.block_sparse_moe.gate.weight"])

            tensors[f"model.layers.{l}.mlp.w1.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight"]) for e in range(config["num_local_experts"])])
            tensors[f"model.layers.{l}.mlp.w2.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]) for e in range(config["num_local_experts"])])
            tensors[f"model.layers.{l}.mlp.w3.weight"] = torch.stack([conv(weights[f"model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight"]) for e in range(config["num_local_experts"])])
        elif arch in ["minicpm"] and "num_experts" in config:
            tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"model.layers.{l}.mlp.gate.weight"])

            tensors[f"model.layers.{l}.mlp.w1.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.w1.weight"]) for e in range(config["num_experts"])])
            tensors[f"model.layers.{l}.mlp.w2.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.w2.weight"]) for e in range(config["num_experts"])])
            tensors[f"model.layers.{l}.mlp.w3.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.w3.weight"]) for e in range(config["num_experts"])])
        elif arch in ["olmoe"]:
            tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"model.layers.{l}.mlp.gate.weight"])

            tensors[f"model.layers.{l}.mlp.w1.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"]) for e in range(config["num_experts"])])
            tensors[f"model.layers.{l}.mlp.w2.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"]) for e in range(config["num_experts"])])
            tensors[f"model.layers.{l}.mlp.w3.weight"] = torch.stack([conv(weights[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"]) for e in range(config["num_experts"])])
        else:
            tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
            tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
            tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    if config.get("tie_word_embeddings", None) != True:
        tensors["model.output.weight"] = conv(weights["lm_head.weight"])
elif arch == "internlm2":
    tensors["model.embed.weight"] = conv(weights["model.tok_embeddings.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.attention_norm.weight"].float()

        head_dim = metadata["head_dim"]
        n_heads = config["num_attention_heads"]
        n_kv_heads = config.get("num_key_value_heads", n_heads)
        kv_mul = n_heads // n_kv_heads

        wqkv = weights[f"model.layers.{l}.attention.wqkv.weight"]
        wqkv = wqkv.unflatten(0, (n_kv_heads, kv_mul + 2, head_dim))

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(wqkv[:, :kv_mul].flatten(0, 2), n_heads, head_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(wqkv[:, kv_mul].flatten(0, 1), n_kv_heads, head_dim))

        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(wqkv[:, kv_mul+1].flatten(0, 1))
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.attention.wo.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.ffn_norm.weight"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.feed_forward.w1.weight"])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.feed_forward.w2.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.feed_forward.w3.weight"])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    tensors["model.output.weight"] = conv(weights["output.weight"])
elif arch == "olmo":
    tensors["model.embed.weight"] = conv(weights["model.transformer.wte.weight"])

    for l in range(config["n_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = torch.ones(config["d_model"], dtype=torch.float32)

        dim = config["d_model"]
        head_dim = dim // config["n_heads"]
        hidden_dim = (config["mlp_hidden_size"] or config["d_model"] * config["mlp_ratio"]) // 2

        attn_proj = weights[f"model.transformer.blocks.{l}.att_proj.weight"]
        assert attn_proj.shape == (dim * 3, dim)

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(attn_proj[:dim], config["n_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(attn_proj[dim:dim*2], config["n_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(attn_proj[dim*2:])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.transformer.blocks.{l}.attn_out.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = torch.ones(config["d_model"], dtype=torch.float32)

        mlp_proj = weights[f"model.transformer.blocks.{l}.ff_proj.weight"]
        assert mlp_proj.shape == (hidden_dim * 2, dim)

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(mlp_proj[hidden_dim:])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.transformer.blocks.{l}.ff_out.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(mlp_proj[:hidden_dim])

    tensors["model.norm.weight"] = torch.ones(config["d_model"], dtype=torch.float32)
    if not config["weight_tying"]:
        tensors["model.output.weight"] = conv(weights["model.transformer.ff_out.weight"])
elif arch == "dbrx":
    tensors["model.embed.weight"] = conv(weights["transformer.wte.weight"])

    for l in range(config["n_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"transformer.blocks.{l}.norm_attn_norm.norm_1.weight"].float()

        head_dim = config["d_model"] // config["n_heads"]
        n_heads = config["n_heads"]
        n_kv_heads = config["attn_config"]["kv_n_heads"]

        dim = config["d_model"]
        hidden_dim = config["ffn_config"]["ffn_hidden_size"]
        n_experts = config["ffn_config"]["moe_num_experts"]

        wqkv = weights[f"transformer.blocks.{l}.norm_attn_norm.attn.Wqkv.weight"]

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(wqkv[:n_heads*head_dim], n_heads, head_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(wqkv[n_heads*head_dim:(n_heads+n_kv_heads)*head_dim], n_kv_heads, head_dim))
        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(wqkv[(n_heads+n_kv_heads)*head_dim:])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"transformer.blocks.{l}.norm_attn_norm.attn.out_proj.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"transformer.blocks.{l}.norm_attn_norm.norm_2.weight"].float()

        tensors[f"model.layers.{l}.moegate.weight"] = conv(weights[f"transformer.blocks.{l}.ffn.router.layer.weight"])

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"transformer.blocks.{l}.ffn.experts.mlp.w1"].view(n_experts, hidden_dim, dim))
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"transformer.blocks.{l}.ffn.experts.mlp.w2"].view(n_experts, hidden_dim, dim).transpose(1, 2).contiguous())
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"transformer.blocks.{l}.ffn.experts.mlp.v1"].view(n_experts, hidden_dim, dim))

    tensors["model.norm.weight"] = weights["transformer.norm_f.weight"].float()
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
elif arch == "phi3":
    tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

        head_dim = config["hidden_size"] // config["num_attention_heads"]
        n_heads = config["num_attention_heads"]
        n_kv_heads = config.get("num_key_value_heads", n_heads)

        wqkv = weights[f"model.layers.{l}.self_attn.qkv_proj.weight"]

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(wqkv[:n_heads*head_dim], n_heads, head_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(wqkv[n_heads*head_dim:(n_heads+n_kv_heads)*head_dim], n_kv_heads, head_dim))

        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(wqkv[(n_heads+n_kv_heads)*head_dim:])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

        hidden_dim = config["intermediate_size"]

        mlp_proj = weights[f"model.layers.{l}.mlp.gate_up_proj.weight"]

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(mlp_proj[:hidden_dim])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(mlp_proj[hidden_dim:])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])

# add tokenizer tensors at the end (to maximize the chance of model tensor alignment)
# note: we concatenate all bytes of all tokens into a single tensor
tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])
tensors["tokenizer.scores"] = torch.tensor(scores, dtype=torch.float32)

print(f"\rSaving {len(tensors)} tensors..." + " " * 40)

# in a perfect world, we would just use HF safetensors.torch.save_file
# however, not only does it not support fp8 (https://github.com/huggingface/safetensors/pull/404), it also copies every tensor
# our models are large, so we'll implement a custom save function. could even materialize converted tensors lazily later.
def save_file(tensors, filename, metadata=None):
    _TYPES = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
    }
    _ALIGN = 256

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = metadata
    for k, v in tensors.items():
        size = v.numel() * v.element_size()
        header[k] = { "dtype": _TYPES[v.dtype], "shape": v.shape, "data_offsets": [offset, offset + size] }
        offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(len(hjson).to_bytes(8, byteorder="little"))
        f.write(hjson)
        for k, v in tensors.items():
            assert v.layout == torch.strided and v.is_contiguous()
            v.view(torch.uint8).numpy().tofile(f)

# metadata values must be strings in safetensors
save_file(tensors, args.output, {k: str(v) for k, v in metadata.items()})
