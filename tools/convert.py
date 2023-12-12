# Produce a safetensors model file out of multiple inputs
# python convert.py merged.safetensors --models file1.safetensors file2.safetensors ...

import argparse
import safetensors
import safetensors.torch

args = argparse.ArgumentParser()
args.add_argument("output", type=str)
args.add_argument("--models", type=str, nargs="+", required=True)
args = args.parse_args()

tensors = {}
for fn in args.models:
    with safetensors.safe_open(fn, framework="pt") as f:
        for k in f.keys():
            assert(k not in tensors)
            tensors[k] = f.get_tensor(k)

# huggingface permutes WQ and WK, this function reverses it
# see https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
def permute_reverse(w, heads, dim1, dim2):
    return w.view(heads, 2, dim1 // heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

for k, v in tensors.items():
    if "self_attn.q_proj" in k or "self_attn.k_proj" in k:
        n_heads = 32
        dim = v.shape[0]
        v = permute_reverse(v, n_heads, dim, dim)

    # convert to f32 for now
    tensors[k] = v.float()

safetensors.torch.save_file(tensors, args.output)
