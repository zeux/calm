# 😌 calm

This is an implementation of language model inference, aiming to get maximum single-GPU single-batch hardware utilization for LLM architectures with a minimal implementation and no dependencies[^1].

The goal of this project is experimentation and prototyping; it does not aim to be production ready or stable.

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
- Mixtral (Llama-like with FFN in every layer replaced by a mixture of experts)

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
- Mixtral architecture
  - Mixtral 8x7B (mistralai/Mixtral-8x7B-Instruct-v0.1)

## Supported formats

Model weights support `fp16`, `fp8` and `gf4` formats; the weight type is specified at conversion time via `--dtype` argument to `convert.py`, and defaults to `fp8`.

`fp16` corresponds to 16-bit floating point (e5m10). Note that some models store weights in bf16 which will be automatically converted.

`fp8` corresponds to 8-bit floating point (e5m2). Using `fp8` carries a ~0.5% perplexity penalty at almost double the inference speed and half the model size. e4m3 variant of `fp8` would result in a much smaller perplexity penalty (~0.1%) with basic tensor scaling, but it's currently not used because of performance issues wrt floating-point conversion.

`gf4` corresponds to 4-bit grouped floating point (8 values are stored in 32 bits using 3 bit quantized scale per value and one fp8 group scale). Using `gf4` currently carries a ~5% perplexity penalty but increases inference speed by ~75% and halves the model size compared to `fp8`. Quantization code is currently naive and further improvements are planned. Unlike llama.cpp's K-quants, `gf4` quantization is pure and uniform - all layers are quantized to exactly 4 bits per weight.

KV cache is using `fp16`.

## Performance

Auto-regressive prediction for a single sequence needs to read the entire model and the entire KV cache (until current token) for every token. As such, given an optimal implementation we'd expect the process to be bandwidth bound. Note that the cost of token generation at the beginning of the sequence should be smaller than the cost at the end of the sequence due to the need to read data from KV cache.

When using NVidia GeForce RTX 4090, `calm` gets the following performance on a few models; each model is measured with `fp16`, `fp8` and `gf4` weights at the beginning of the context window (first 32 tokens) and at the end (last 32 tokens with an offset 2000 for 2048 contexts and 4000 for 4096 contexts):

| Model (context)     | Performance (first 32 tokens) | Performance (last 32 tokens) |
| ----------- | ----------- | ----------- |
| Llama2 7B (4096), fp16 | 67 tok/s (905 GB/s) | 58 tok/s (910 GB/s) |
| Llama2 7B (4096), fp8 | 126 tok/s (850 GB/s) | 98 tok/s (870 GB/s) |
| Llama2 7B (4096), gf4 | 222 tok/s (750 GB/s) | 148 tok/s (815 GB/s) |
| Llama2 13B (4096), fp8 | 67 tok/s (879 GB/s) | 54 tok/s (895 GB/s) |
| Llama2 13B (4096), gf4 | 123 tok/s (802 GB/s) | 86 tok/s (847 GB/s) |
| Mistral 7B (4096), fp16 | 63 tok/s (914 GB/s) | 61 tok/s (914 GB/s) |
| Mistral 7B (4096), fp8 | 119 tok/s (865 GB/s) | 111 tok/s (869 GB/s) |
| Mistral 7B (4096), gf4 | 214 tok/s (778 GB/s) | 190 tok/s (790 GB/s) |
| Phi 2.7B (2048), fp16 | 167 tok/s (932 GB/s) | 149 tok/s (930 GB/s) |
| Phi 2.7B (2048), fp8 | 313 tok/s (874 GB/s) | 256 tok/s (881 GB/s) |
| Phi 2.7B (2048), gf4 | 551 tok/s (771 GB/s) | 395 tok/s (811 GB/s) |
| Mixtral 8x7B (4096), gf4 | 127 tok/s (820 GB/s) | 118 tok/s (822 GB/s) |
| Yi 34B (4096), gf4 | 49 tok/s (846 GB/s) | 46 tok/s (837 GB/s) |

Currently prompts are processed serially, one token at a time; in the future, prompt processing will need to be parallelized to avoid the bandwidth bottleneck.

With smaller weights on small models, getting closer to bandwidth limit becomes more difficult. Future optimizations may increase the gap here for small models, although smaller weights are most valuable to be able to infer larger models.

RTX 4090 has a peak bandwidth of ~1008 GB/s, however it's unclear if a peak higher than ~950 GB/s is attainable in practice[^3]. The code has not been heavily tuned for datacenter-grade hardware (A100/H100) or earlier NVidia architectures yet.

## Model files

calm uses [🤗 Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model hyperparameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

[^1]: CUDA runtime and compiler is used for GPU acceleration, but no CUDA or C libraries are used. Python conversion scripts use safetensors and torch, see `tools/requirements.txt`.
[^2]: Linux is the only supported OS at the moment.
[^3]: Based on testing a specific Gigabyte GeForce RTX 4090 where both individual kernels from this repository and cuBLAS peak at about ~955 GB/s.
