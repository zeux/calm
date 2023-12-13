# Produce a safetensors model file out of multiple inputs
# python convert.py merged.safetensors --models file1.safetensors file2.safetensors ...

import argparse
import os.path
import safetensors
import safetensors.torch
import torch

args = argparse.ArgumentParser()
args.add_argument("output", type=str)
args.add_argument("--models", type=str, nargs="+", required=True)
args = args.parse_args()

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

for k, v in tensors.items():
    if "self_attn.q_proj" in k or "self_attn.k_proj" in k:
        n_heads = 32
        dim1 = v.shape[0]
        dim2 = v.shape[1]
        v = permute_reverse(v, n_heads // (dim2 // dim1), dim1, dim2)

    # convert to f32 for now
    tensors[k] = v.float()

safetensors.torch.save_file(tensors, args.output)
