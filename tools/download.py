# Download model folder from HuggingFace
# python download.py model_folder repo_id

import argparse
import huggingface_hub

argp = argparse.ArgumentParser()
argp.add_argument("output", type=str)
argp.add_argument("repo", type=str)
argp.add_argument("--all", action="store_true")
args = argp.parse_args()

# download model folder from HuggingFace, excluding .bin files (assume the model contains safetensors)
ignore_patterns = ["*.bin", "*.pth", "*.pt", "*.gguf", "consolidated.safetensors"] if not args.all else []
huggingface_hub.snapshot_download(repo_id=args.repo, local_dir=args.output, ignore_patterns=ignore_patterns)
