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

for k, v in tensors.items():
    # convert to f32 for now
    tensors[k] = v.float()

safetensors.torch.save_file(tensors, args.output)
