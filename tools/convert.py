# Produce a safetensors model file out of multiple inputs
# python convert.py model.safetensors --config config.json --models file1.bin file2.bin ...

import argparse
import json
import os.path
import safetensors
import safetensors.torch
import torch

args = argparse.ArgumentParser()
args.add_argument("output", type=str)
args.add_argument("--config", type=str, required=True)
args.add_argument("--models", type=str, nargs="+", required=True)
args.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
args = args.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

metadata = {}

# hardcoded in C
assert config["hidden_act"] == "silu"
assert config["bos_token_id"] == 1
assert config["eos_token_id"] == 2
assert config["rms_norm_eps"] == 1e-5

# customizable
metadata["dim"] = config["hidden_size"]
metadata["hidden_dim"] = config["intermediate_size"]
metadata["n_layers"] = config["num_hidden_layers"]
metadata["n_heads"] = config["num_attention_heads"]
metadata["n_kv_heads"] = config["num_key_value_heads"]
metadata["vocab_size"] = config["vocab_size"]
if "rope_theta" in config:
    metadata["rope_theta"] = config["rope_theta"]

# load model files
tensors = {}
for fn in args.models:
    ext = os.path.splitext(fn)[1]
    if ext == ".safetensors":
        with safetensors.safe_open(fn, framework="pt") as f:
            for k in f.keys():
                assert(k not in tensors)
                tensors[k] = f.get_tensor(k)
    elif ext == ".bin":
        weights = torch.load(fn, weights_only=True)
        for k in weights.keys():
            assert(k not in tensors)
            tensors[k] = weights[k]
    else:
        raise Exception("Unknown file extension: {}; expected .safetensors or .bin".format(ext))

# huggingface permutes WQ and WK, this function reverses it
# see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def permute_reverse(w, heads, dim1, dim2):
    return w.view(heads, 2, dim1 // heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

# convert tensors to dtype and permute if necessary
for k, v in tensors.items():
    if "self_attn.q_proj" in k or "self_attn.k_proj" in k:
        n_heads = 32
        dim1 = v.shape[0]
        dim2 = v.shape[1]
        v = permute_reverse(v, n_heads // (dim2 // dim1), dim1, dim2)

    tensors[k] = v.to(dtype)

# metadata values must be strings in safetensors
safetensors.torch.save_file(tensors, args.output, {k: str(v) for k, v in metadata.items()})
