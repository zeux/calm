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
argp.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
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
            argp.error("no tokenizer.model found in {}".format(args.input))
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

# load tokenizer model
tokens, scores = [], []

ext = os.path.splitext(args.tokenizer)[1]
if ext == ".model":
    sp_model = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    assert sp_model.vocab_size() == sp_model.get_piece_size()
    assert sp_model.vocab_size() == config["vocab_size"]
    assert sp_model.bos_id() == config["bos_token_id"]
    assert sp_model.eos_id() == config["eos_token_id"]

    # get all the tokens (postprocessed) and their scores as floats
    for i in range(sp_model.vocab_size()):
        t = sp_model.id_to_piece(i)
        s = sp_model.get_score(i)
        t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')
        assert b.count(0) == 0 # no null bytes allowed

        tokens.append(b)
        scores.append(s)
else:
    raise Exception("Unknown tokenizer file extension: {}; expected .model".format(ext))

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
def permute_reverse(w, heads, dim1, dim2):
    return w.view(heads, 2, dim1 // heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

# convert weights to dtype and permute if necessary
for k, v in weights.items():
    if "self_attn.q_proj" in k or "self_attn.k_proj" in k:
        n_heads = 32
        dim1 = v.shape[0]
        dim2 = v.shape[1]
        v = permute_reverse(v, n_heads // (dim2 // dim1), dim1, dim2)

    tensors[k] = v.to(dtype)

# metadata values must be strings in safetensors
safetensors.torch.save_file(tensors, args.output, {k: str(v) for k, v in metadata.items()})
