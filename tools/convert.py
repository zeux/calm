# Produce a safetensors model file out of multiple inputs
# python convert.py model.safetensors --config config.json --models file1.bin file2.bin ...

import argparse
import json
import os.path
import safetensors
import safetensors.torch
import sentencepiece
import torch

argp = argparse.ArgumentParser()
argp.add_argument("output", type=str)
argp.add_argument("input", type=str, nargs="?")
argp.add_argument("--config", type=str)
argp.add_argument("--tokenizer", type=str)
argp.add_argument("--models", type=str, nargs="+")
argp.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp8"])
args = argp.parse_args()

if args.input is not None:
    # assume input is a directory with HuggingFace layout
    if args.config is None:
        args.config = os.path.join(args.input, "config.json")
        if not os.path.exists(args.config):
            argp.error("no config.json found in {}".format(args.input))
    if args.tokenizer is None:
        args.tokenizer = os.path.join(args.input, "tokenizer.model")
        if not os.path.exists(args.tokenizer):
            args.tokenizer = os.path.join(args.input, "tokenizer.json")
        if not os.path.exists(args.tokenizer):
            argp.error("no tokenizer.model or tokenizer.json found in {}".format(args.input))
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
arch_remap = {"LlamaForCausalM": "llama", "MistralForCausalLM": "mistral", "PhiForCausalLM": "phi"}
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
elif arch == "phi":
    # hardcoded in C
    assert config["activation_function"] == "gelu_new"
    assert config["layer_norm_epsilon"] == 1e-5

    # customizable
    metadata["dim"] = config["n_embd"]
    metadata["hidden_dim"] = config["n_inner"] or config["n_embd"] * 4
    metadata["n_layers"] = config["n_layer"]
    metadata["n_heads"] = config["n_head"]
    metadata["n_kv_heads"] = config["n_head_kv"] or config["n_head"]
    metadata["vocab_size"] = config["vocab_size"]
    metadata["max_seq_len"] = config["n_positions"]
    metadata["bos_token_id"] = -1
    metadata["eos_token_id"] = 50256 # todo: read from tokenizer_config
    metadata["rope_theta"] = 10000.0 # hardcoded in model

# load tokenizer model
tokens = [""] * config["vocab_size"]
scores = [0] * config["vocab_size"]

ext = os.path.splitext(args.tokenizer)[1]
if ext == ".model":
    sp_model = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    assert sp_model.vocab_size() <= config["vocab_size"]
    assert sp_model.bos_id() == config["bos_token_id"]
    assert sp_model.eos_id() == config["eos_token_id"]

    for i in range(sp_model.vocab_size()):
        tokens[i] = sp_model.id_to_piece(i)
        scores[i] = sp_model.get_score(i)
elif ext == ".json":
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
else:
    raise Exception("Unknown tokenizer file extension: {}; expected .model".format(ext))

# postprocess tokens
for i, t in enumerate(tokens):
    t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
    t = t.replace('\u0120', ' ') # some gpt-based tokenizers use this character as whitespace
    t = '\n' if t == '\u010a' else t  # some gpt-based tokenizers use this character as newline

    b = t.encode('utf-8')
    assert b.count(0) == 0 # no null bytes allowed

    tokens[i] = b

# add tokenizer tensors
# note: we concatenate all bytes of all tokens into a single tensor
tensors["tokenizer.tokens"] = torch.cat([torch.tensor([x for x in b] + [0], dtype=torch.uint8) for b in tokens])
tensors["tokenizer.scores"] = torch.tensor(scores, dtype=torch.float32)

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
        pth = torch.load(fn, weights_only=True)
        for k in pth.keys():
            assert(k not in weights)
            weights[k] = pth[k]
    else:
        raise Exception("Unknown model file extension: {}; expected .safetensors or .bin".format(ext))

# huggingface permutes WQ and WK, this function reverses it
# see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def permute_reverse(w, heads):
    dim1 = w.shape[0]
    dim2 = w.shape[1]
    return w.view(heads, 2, dim1 // heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

# fp8 support requires torch 2.1, but we support other dtypes on earlier versions
dtype = {"fp16": torch.float16, "fp8": getattr(torch, "float8_e5m2", None)}[args.dtype]
assert dtype

# convert weights
def conv(t):
    return t.to(dtype)

tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

for l in range(config["num_hidden_layers"]):
    tensors[f"model.layers.{l}.attn.norm.weight"] = weights[f"model.layers.{l}.input_layernorm.weight"].float()
    tensors[f"model.layers.{l}.attn.wq.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.q_proj.weight"], config["num_attention_heads"]))
    tensors[f"model.layers.{l}.attn.wk.weight"] = conv(permute_reverse(weights[f"model.layers.{l}.self_attn.k_proj.weight"], config["num_key_value_heads"]))
    tensors[f"model.layers.{l}.attn.wv.weight"] = conv(weights[f"model.layers.{l}.self_attn.v_proj.weight"])
    tensors[f"model.layers.{l}.attn.wo.weight"] = conv(weights[f"model.layers.{l}.self_attn.o_proj.weight"])

    tensors[f"model.layers.{l}.mlp.norm.weight"] = weights[f"model.layers.{l}.post_attention_layernorm.weight"].float()

    tensors[f"model.layers.{l}.mlp.w1.weight"] = conv(weights[f"model.layers.{l}.mlp.gate_proj.weight"])
    tensors[f"model.layers.{l}.mlp.w2.weight"] = conv(weights[f"model.layers.{l}.mlp.down_proj.weight"])
    tensors[f"model.layers.{l}.mlp.w3.weight"] = conv(weights[f"model.layers.{l}.mlp.up_proj.weight"])

tensors["model.norm.weight"] = weights["model.norm.weight"].float()
tensors["model.output.weight"] = conv(weights["lm_head.weight"])

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

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = metadata
    for k, v in tensors.items():
        size = v.numel() * v.element_size()
        header[k] = { "dtype": _TYPES[v.dtype], "shape": v.shape, "data_offsets": [offset, offset + size] }
        offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-len(hjson) % 8)

    with open(filename, "wb") as f:
        f.write(len(hjson).to_bytes(8, byteorder="little"))
        f.write(hjson)
        for k, v in tensors.items():
            assert v.layout == torch.strided and v.is_contiguous()
            v.view(torch.uint8).numpy().tofile(f)

# metadata values must be strings in safetensors
# save_file = safetensors.torch.save_file
save_file(tensors, args.output, {k: str(v) for k, v in metadata.items()})
