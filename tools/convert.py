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
arch_remap = {"LlamaForCausalLM": "llama", "MistralForCausalLM": "mistral", "PhiForCausalLM": "phi", "QWenLMHeadModel": "qwen"}
assert arch in arch_remap, "Unsupported architecture: {}; must be one of: {}".format(arch, list(arch_remap.keys()))
arch = arch_remap[arch]

metadata["arch"] = arch
metadata["dtype"] = args.dtype

if arch in ["llama", "mistral"]:
    # hardcoded in C
    assert config["hidden_act"] == "silu"
    assert config["rms_norm_eps"] == 1e-5

    # customizable
    metadata["dim"] = config["hidden_size"]
    metadata["hidden_dim"] = config["intermediate_size"]
    metadata["n_layers"] = config["num_hidden_layers"]
    metadata["n_heads"] = config["num_attention_heads"]
    metadata["n_kv_heads"] = config["num_key_value_heads"]
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = config["max_position_embeddings"]
    metadata["bos_token_id"] = config["bos_token_id"]
    metadata["eos_token_id"] = config["eos_token_id"]
    metadata["rope_theta"] = config.get("rope_theta", 10000.0)
    metadata["rotary_dim"] = config["hidden_size"] // config["num_attention_heads"]
elif arch == "qwen":
    # customizable
    metadata["dim"] = config["hidden_size"]
    metadata["hidden_dim"] = config["intermediate_size"] // 2
    metadata["n_layers"] = config["num_hidden_layers"]
    metadata["n_heads"] = config["num_attention_heads"]
    metadata["n_kv_heads"] = config["num_attention_heads"]
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = config["seq_length"]
    metadata["bos_token_id"] = -1
    metadata["eos_token_id"] = 151643 # <|endoftext|> hardcoded in tokenization_qwen.py
    metadata["rope_theta"] = config.get("rope_theta", 10000.0)
    metadata["rotary_dim"] = config["hidden_size"] // config["num_attention_heads"]
elif arch == "phi":
    # hardcoded in C
    assert config["hidden_act"] == "gelu_new"
    assert config["layer_norm_eps"] == 1e-5

    # customizable
    metadata["dim"] = config["hidden_size"]
    metadata["hidden_dim"] = config["intermediate_size"]
    metadata["n_layers"] = config["num_hidden_layers"]
    metadata["n_heads"] = config["num_attention_heads"]
    metadata["n_kv_heads"] = config["num_key_value_heads"] or config["num_attention_heads"]
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = config["max_position_embeddings"]
    metadata["bos_token_id"] = -1
    metadata["eos_token_id"] = config["eos_token_id"] or 50256 # todo: read from tokenizer_config
    metadata["rope_theta"] = config.get("rope_theta", 10000.0)
    metadata["rotary_dim"] = int(config["hidden_size"] / config["num_attention_heads"] * config["partial_rotary_factor"])

# load tokenizer model
tokens = [""] * config["vocab_size"]
scores = [0] * config["vocab_size"]

ext = os.path.splitext(args.tokenizer)[1]
if ext == ".json":
    with open(args.tokenizer, "r") as f:
        tokenizer = json.load(f)

    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= config["vocab_size"]

    for t, i in vocab.items():
        tokens[i] = t

    # compute score as negative merge index so that earlier merges get selected first
    for i, m in enumerate(tokenizer["model"]["merges"]):
        t1, t2 = m.split(" ")
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
for i, t in enumerate(tokens):
    t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
    t = t.replace('\u0120', ' ') # some gpt-based tokenizers use this character as whitespace
    t = '\n' if t == '\u010a' else t  # some gpt-based tokenizers use this character as newline

    b = t.encode('utf-8')
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
    if torch.cuda.is_available(): t = t.cuda()
    # groups of 8 values
    gt = t.unflatten(-1, (-1, 8))
    # max (abs) of each group
    _, gmaxi = gt.abs().max(-1)
    gmax = gt.gather(-1, gmaxi.unsqueeze(-1))
    # round gmax to fp8 to make sure we're quantizing to the right range
    gmax = gmax.to(torch.float8_e5m2).to(gmax.dtype)
    # normalize each group by -max ([-1, 1]) and quantize to [0, 8)
    # note that 8 needs to be clamped to 7 since positive half of the range is shorter
    gtq = ((gt / gmax).to(torch.float16) * -4 + 4).clamp(0, 7).round().to(torch.int32)
    # assemble the results
    gtq <<= torch.tensor([8 + i * 3 for i in range(8)], dtype=torch.int32, device=gtq.device)
    gtr = gtq.sum(-1, dtype=torch.int32)
    gtr += gmax.squeeze(-1).to(torch.float8_e5m2).view(torch.uint8)
    return gtr.cpu()

# convert weights
progress = 0
def conv(t):
    global progress
    progress += 1
    print(f"\rConverting tensor {progress}: {t.shape}", end="", flush=True)
    return gf4(t) if dtype == torch.uint8 else t.to(dtype)

if arch in ["llama", "mistral"]:
    tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()

        head_dim = config["hidden_size"] // config["num_attention_heads"]

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], config["num_attention_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], config["num_key_value_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
elif arch == "qwen":
    tensors["model.embed.weight"] = conv(weights["transformer.wte.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"transformer.h.{l}.ln_1.weight"].float()

        dim = config["hidden_size"]
        head_dim = dim // config["num_attention_heads"]

        wkv_w = weights[f"transformer.h.{l}.attn.c_attn.weight"]
        wkv_b = weights[f"transformer.h.{l}.attn.c_attn.bias"]
        assert wkv_w.shape[0] == 3 * dim and wkv_b.shape[0] == 3 * dim

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(wkv_w[:dim], config["num_attention_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wq.bias"] = permute_reverse(wkv_b[:dim], config["num_attention_heads"], head_dim).float()
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(wkv_w[dim:dim*2], config["num_attention_heads"], head_dim))
        tensors[f"model.layers.{l}.attn.wk.bias"] = permute_reverse(wkv_b[dim:dim*2], config["num_attention_heads"], head_dim).float()
        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(wkv_w[dim*2:])
        tensors[f"model.layers.{l}.attn.wv.bias"] = wkv_b[dim*2:].float()
        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"transformer.h.{l}.attn.c_proj.weight"])

        tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"transformer.h.{l}.ln_2.weight"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"transformer.h.{l}.mlp.w2.weight"])
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"transformer.h.{l}.mlp.c_proj.weight"])
        tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"transformer.h.{l}.mlp.w1.weight"])

    tensors["model.norm.weight"] = weights["transformer.ln_f.weight"].float()
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
elif arch == "phi":
    tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

    for l in range(config["num_hidden_layers"]):
        tensors[f"model.layers.{l}.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()
        tensors[f"model.layers.{l}.norm.bias"] = weights[f"model.layers.{l}.input_layernorm.bias"].float()

        dim = config["hidden_size"]
        rotary_dim = metadata["rotary_dim"]

        tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], config["num_attention_heads"], rotary_dim))
        tensors[f"model.layers.{l}.attn.wq.bias"] = permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.bias"], config["num_attention_heads"], rotary_dim).float()
        tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], config["num_attention_heads"], rotary_dim))
        tensors[f"model.layers.{l}.attn.wk.bias"] = permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.bias"], config["num_attention_heads"], rotary_dim).float()
        tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
        tensors[f"model.layers.{l}.attn.wv.bias"] = weights[f"model.layers.{l}.self_attn.v_proj.bias"].float()

        tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.dense.weight"])
        tensors[f"model.layers.{l}.attn.wo.bias"] = weights[f"model.layers.{l}.self_attn.dense.bias"].float()

        tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.fc1.weight"])
        tensors[f"model.layers.{l}.mlp.w1.bias"] = weights[f"model.layers.{l}.mlp.fc1.bias"].float()
        tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.fc2.weight"])
        tensors[f"model.layers.{l}.mlp.w2.bias"] = weights[f"model.layers.{l}.mlp.fc2.bias"].float()

    tensors["model.norm.weight"] = weights["model.final_layernorm.weight"].float()
    tensors["model.norm.bias"] = weights["model.final_layernorm.bias"].float()
    tensors["model.output.weight"] = conv(weights["lm_head.weight"])
    tensors["model.output.bias"] = weights["lm_head.bias"].float()

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
# save_file = safetensors.torch.save_file
save_file(tensors, args.output, {k: str(v) for k, v in metadata.items()})
