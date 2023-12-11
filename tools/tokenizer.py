# Produce a binary tokenizer model file out of sentencepiece model
# python tokenizer.py tokenizer.bin --model tokenizer.model

# Based on source from llama2.c which is based on Llama2 source which is copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
import argparse
from typing import List

from sentencepiece import SentencePieceProcessor

args = argparse.ArgumentParser()
args.add_argument("output", type=str)
args.add_argument("--model", type=str, required=True)
args = args.parse_args()

sp_model = SentencePieceProcessor(model_file=args.model)
assert sp_model.vocab_size() == sp_model.get_piece_size()

# C code currently assumes these are hardcoded
assert sp_model.bos_id() == 1
assert sp_model.eos_id() == 2

# get all the tokens (postprocessed) and their scores as floats
tokens, scores = [], []
for i in range(sp_model.vocab_size()):
    # decode the token and light postprocessing
    t = sp_model.id_to_piece(i)
    s = sp_model.get_score(i)
    if i == sp_model.bos_id():
        t = '\n<s>\n'
    elif i == sp_model.eos_id():
        t = '\n</s>\n'
    t = t.replace('\u2581', ' ') # sentencepiece uses this character as whitespace
    b = t.encode('utf-8') # bytes of this token, utf-8 encoded

    tokens.append(b)
    scores.append(s)

# record the max token length
max_token_length = max(len(t) for t in tokens)

# write to a binary file
# the tokenizer.bin file is the same as .model file, but .bin
with open(args.output, 'wb') as f:
    f.write(struct.pack("I", max_token_length))
    for bytes, score in zip(tokens, scores):
        f.write(struct.pack("fI", score, len(bytes)))
        f.write(bytes)
