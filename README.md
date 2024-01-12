# 😌 calm

This is an implementation of language model inference, aiming to get maximum single-GPU single-batch hardware utilization for LLM architectures with a minimal implementation and no dependencies[^1].

The goal of this project is experimentation and prototyping; it does not aim to be production ready or stable. It is heavily work in progress.

If you need support for a wide range of models, computing devices or quantization methods, you're probably looking for [llama.cpp](https://github.com/ggerganov/llama.cpp) or [🤗 Transformers](https://github.com/huggingface/transformers). If you need to run inference for multiple batches, you're probably looking for [vLLM](https://github.com/vllm-project/vllm).

Parts of this code are based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Running

To build and run `calm`, you need to download and convert a model, build the code using `make`[^2] and run it:

```sh
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python tools/convert.py mistral-7b-instruct.calm Mistral-7B-Instruct-v0.2/
make && ./build/run mistral-7b-instruct.calm -i "Q: What is the meaning of life?" -t 0
```

Before running Python you may want to install the dependencies via `pip install -r tools/requirements.txt`. When using git, git-lfs is required and the download size may be larger than necessary; you can use `tools/download.py` instead:

```sh
python tools/download.py Mistral-7B-Instruct-v0.2/ mistralai/Mistral-7B-Instruct-v0.2
```

## Supported models

calm currently supports the following model architectures:

- Llama-like (RMSNorm normalization, SiLU activation, sequential attention mixing and FFN, RoPE)
- Phi (LayerNorm normalization, GELU activation, parallel attention mixing, partial RoPE)

It has been tested on following models:

- Llama architecture
  - TinyLlama 1.1B (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  - Llama2 7B (meta-llama/Llama-2-7b-chat-hf)
  - Llama2 13B (meta-llama/Llama-2-13b-chat-hf)
  - LLaMA Pro 8B (TencentARC/LLaMA-Pro-8B-Instruct)
  - Yi 34B (01-ai/Yi-34B-Chat)
- Mistral architecture
  - Mistral 7B (mistralai/Mistral-7B-Instruct-v0.2)
  - SOLAR 10.7B (upstage/SOLAR-10.7B-Instruct-v1.0)
- Qwen architecture
  - Qwen 7B (Qwen/Qwen-7B-Chat)
  - Qwen 14B (Qwen/Qwen-14B-Chat)
- Phi architecture
  - Phi1.5 (microsoft/phi-1_5)
  - Phi2 (microsoft/phi-2)

## Supported formats

Model weights support `fp16` and `fp8` formats; the weight type is specified at conversion time via `--dtype` argument to `convert.py`, and defaults to `fp8`.

`fp16` corresponds to 16-bit floating point (e5m10). Note that some models store weights in bf16 which will be automatically converted.

`fp8` corresponds to 8-bit floating point (e5m2). Using `fp8` carries a ~0.5% perplexity penalty at almost double the inference speed and half the model size. e4m3 variant of `fp8` would result in a much smaller perplexity penalty (~0.1%) with basic tensor scaling, but it's currently not used because of performance issues wrt floating-point conversion.

KV cache is using `fp16`.

## Performance

Auto-regressive prediction for a single sequence needs to read the entire model and the entire KV cache (until current token) for every token. As such, given an optimal implementation we'd expect the process to be bandwidth bound. Note that the cost of token generation at the beginning of the sequence should be smaller than the cost at the end of the sequence due to the need to read data from KV cache.

When using NVidia GeForce RTX 4090, `calm` gets the following performance on a few models; each model is measured with `fp16` and `fp8` weights at the beginning of the context window (first 32 tokens) and at the end (last 32 tokens with an offset 2000 for 2048 contexts and 4000 for 4096 contexts):

| Model (context)     | Performance (first 32 tokens) | Performance (last 32 tokens)
| ----------- | ----------- | ----------- |
| Llama2 7B (4096), fp16 | 66 tok/s (895 GB/s) | 58 tok/s (905 GB/s) |
| Llama2 7B (4096), fp8 | 123 tok/s (833 GB/s) | 97 tok/s (860 GB/s) |
| Mistral 7B (4096), fp16 | 62 tok/s (905 GB/s) | 60 tok/s (902 GB/s) |
| Mistral 7B (4096), fp8 | 117 tok/s (850 GB/s) | 109 tok/s (848 GB/s) |
| Phi 2.7B (2048), fp16 | 165 tok/s (922 GB/s) | 147 tok/s (917 GB/s) |
| Phi 2.7B (2048), fp8 | 307 tok/s (856 GB/s) | 250 tok/s (866 GB/s) |

Currently prompts are processed serially, one token at a time; in the future, prompt processing will need to be parallelized to avoid the bandwidth bottleneck.

Currently weights support `fp16` and `fp8` formats; in the future, 4-bit quantization is planned. This will allow running inference at higher tok/s, however the main metric is bandwidth utilization and the goal is to keep it as close to peak as possible at all supported weight formats.

RTX 4090 has a peak bandwidth of ~1008 GB/s, however it's unclear if a peak higher than ~950 GB/s is attainable in practice[^3]. The code has not been heavily tuned for datacenter-grade hardware (A100/H100) or earlier NVidia architectures yet.

## Model files

calm uses [🤗 Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model hyperparameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

[^1]: CUDA runtime and compiler is used for GPU acceleration, but no CUDA or C libraries are used. Python conversion scripts use safetensors and torch, see `tools/requirements.txt`.
[^2]: Linux is the only supported OS at the moment.
[^3]: Based on testing a specific Gigabyte GeForce RTX 4090 where both individual kernels from this repository and cuBLAS peak at about ~955 GB/s.
