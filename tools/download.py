# Download model folder from HuggingFace
# python download.py model_folder repo_id

import argparse
import huggingface_hub

argp = argparse.ArgumentParser()
argp.add_argument("output", type=str)
argp.add_argument("repo", type=str)
args = argp.parse_args()

# download model folder from HuggingFace, excluding .bin files (assume the model contains safetensors)
huggingface_hub.snapshot_download(repo_id=args.repo, local_dir=args.output, local_dir_use_symlinks=False, resume_download=True, token=True, ignore_patterns="*.bin")
