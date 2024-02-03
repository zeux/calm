import argparse
import base64
import json
import os
import safetensors
import safetensors.torch
import torch
import sentencepiece

def load_configuration(args):
    # Load configuration from args or input directory
    # ...

def load_tokens(args):
    # Load tokens from the tokenizer file
    # ...

def convert_weights(args):
    # Convert weights based on the architecture
    # ...

def permute_reverse(w, heads, rotary_dim):
    # Reverse permutation for certain weight tensors
    # ...

def gf4(t):
    # Quantization for the gf4 dtype
    # ...

def convert_tensor(t, dtype):
    # Convert an individual tensor based on dtype
    # ...

def save_safetensors_file(tensors, metadata, output_file):
    # Save tensors to a .safetensors file
    # ...

def main():
    # Main function to orchestrate the conversion process
    # ...

if __name__ == "__main__":
    main()
