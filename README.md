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

You can also run the model in chat mode (for models like Mistral/Mixtral you might want to increase context size via `-c` from the default 4096):

```sh
make && ./build/run mistral-7b-instruct.calm -y "You are a helpful AI assistant."
```

Before running Python you may want to install the dependencies via `pip install -r tools/requirements.txt`. When using git to download models, git-lfs is required and the download size may be larger than necessary; you can use `tools/download.py` instead (assumes models use Safetensors by default):

```sh
python tools/download.py Mistral-7B-Instruct-v0.2/ mistralai/Mistral-7B-Instruct-v0.2
```

## Supported models

calm supports a subset of decoder-only transformer architectures:

- Llama-like baseline (pre/post normalization, gated FFN, sequential attention mixing and FFN, RoPE)
- RoPE enhancements (partial rotary dimension, independent head dimension)
- SiLU or GELU FFN gate activation
- RMSNorm or LayerNorm* normalization (no bias support)
- Optional minor variations (QKV bias, tied embeddings)
- Optional mixture of experts (with top-k expert selection)

It has been tested on following models:

| Architecture      | Models |
|-------------------|--------|
| Llama | [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [Cosmo 1B](https://huggingface.co/HuggingFaceTB/cosmo-1b), [Llama2 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama2 13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [LLaMA Pro 8B](https://huggingface.co/TencentARC/LLaMA-Pro-8B-Instruct), [Yi 6B](https://huggingface.co/01-ai/Yi-6B-Chat/), [Yi 9B](https://huggingface.co/01-ai/Yi-9B), [Yi 34B](https://huggingface.co/01-ai/Yi-34B-Chat), [H2O Danube 1.8B](https://huggingface.co/h2oai/h2o-danube-1.8b-chat), [DeepSeekMath 7B](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct), [LargeWorldModel 7B 1M](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M) |
| Mistral | [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Mistral Pro 8B](https://huggingface.co/TencentARC/Mistral_Pro_8B_v0.1), [SOLAR 10.7B](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0), [GritLM 7B](https://huggingface.co/GritLM/GritLM-7B) |
| Qwen2 | [Qwen1.5 0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B), [Qwen1.5 1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B), [Qwen1.5 4B](https://huggingface.co/Qwen/Qwen1.5-4B), [Qwen1.5 7B](https://huggingface.co/Qwen/Qwen1.5-7B), [Qwen1.5 14B](https://huggingface.co/Qwen/Qwen1.5-14B) |
| Mixtral | [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [GritLM 8x7B](https://huggingface.co/GritLM/GritLM-8x7B) |
| OLMo    | [OLMo 1B](https://huggingface.co/allenai/OLMo-1B), [OLMo 7B](https://huggingface.co/allenai/OLMo-7B) |
| Gemma   | [Gemma 2B](https://huggingface.co/google/gemma-2b-it), [Gemma 7B](https://huggingface.co/google/gemma-7b-it) (*note: 7B version has issues with fp8 quantization*)  |
| MiniCPM | [MiniCPM 2B](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16) |
| Cohere | [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01) |

## Supported formats

Model weights support `fp16`, `fp8` and `gf4` formats; the weight type is specified at conversion time via `--dtype` argument to `convert.py`, and defaults to `fp8`.

`fp16` corresponds to 16-bit floating point (e5m10). Note that some models store weights in bf16 which will be automatically converted.

`fp8` corresponds to 8-bit floating point (e5m2). Using `fp8` carries a ~0.5% perplexity penalty at almost double the inference speed and half the model size. e4m3 variant of `fp8` would result in a much smaller perplexity penalty (~0.1%) with basic tensor scaling, but it's currently not used because of performance issues wrt floating-point conversion.

`gf4` corresponds to 4-bit grouped floating point (8 values are stored in 32 bits using 3 bit quantized scale per value and one fp8 group scale). Using `gf4` currently carries a perplexity penalty but increases inference speed by ~75% and halves the model size compared to `fp8`. Unlike llama.cpp's K-quants, `gf4` quantization is pure and uniform - all layers are quantized to exactly 4 bits per weight.

KV cache is using `fp16` by default; when using longer contexts (> 4096), CUDA implementation automatically switches to `fp8` to improve memory/performance. This comes at a small perplexity cost.

## Performance

Auto-regressive prediction for a single sequence needs to read the entire model and the entire KV cache (until current token) for every token. As such, given an optimal implementation we'd expect the process to be bandwidth bound. Note that the cost of token generation at the beginning of the sequence should be smaller than the cost at the end of the sequence due to the need to read data from KV cache.

When using NVidia GeForce RTX 4090, `calm` gets the following performance on a few models; each model is measured with `fp16`, `fp8` and `gf4` weights at the beginning of the context window (first 32 tokens) and at the end (last 32 tokens with an offset 2000 for 2048 contexts, 4000 for 4096 contexts and 16000 for 16384 contexts):

| Model (context) | Performance (first 32 tokens) | Performance (last 32 tokens) |
| ----------- | ----------- | ----------- |
| Llama2 7B (4096), fp16 | 69 tok/s (919 GB/s) | 60 tok/s (921 GB/s) |
| Llama2 7B (4096), fp8 | 135 tok/s (893 GB/s) | 103 tok/s (899 GB/s) |
| Llama2 7B (4096), gf4 | 246 tok/s (815 GB/s) | 158 tok/s (857 GB/s) |
| Llama2 13B (4096), fp8 | 70 tok/s (910 GB/s) | 56 tok/s (907 GB/s) |
| Llama2 13B (4096), gf4 | 131 tok/s (848 GB/s) | 88 tok/s (863 GB/s) |
| Mistral 7B (4096), fp16 | 65 tok/s (925 GB/s) | 62 tok/s (916 GB/s) |
| Mistral 7B (4096), fp8 | 127 tok/s (902 GB/s) | 116 tok/s (888 GB/s) |
| Mistral 7B (4096), gf4 | 237 tok/s (843 GB/s) | 203 tok/s (832 GB/s) |
| Mixtral 8x7B (4096), gf4 | 137 tok/s (875 GB/s) | 125 tok/s (862 GB/s) |
| Mixtral 8x7B (16384), gf4 | 137 tok/s (879 GB/s) | 105 tok/s (781 GB/s) |
| Yi 34B (4096), gf4 | 52 tok/s (884 GB/s) | 47 tok/s (851 GB/s) |

Currently prompts are processed serially, one token at a time; in the future, prompt processing will need to be parallelized to avoid the bandwidth bottleneck.

With smaller weights on small models, getting closer to bandwidth limit becomes more difficult. Future optimizations may increase the gap here for small models, although smaller weights are most valuable to be able to infer larger models.

RTX 4090 has a peak bandwidth of ~1008 GB/s, however it's unclear if a peak higher than ~950 GB/s is attainable in practice[^3].

`calm` can run on A100/H100 accelerators (but is mostly tuned for H100 `fp8` weights). When using Mixtral 8x7B (fp8) on 1xH100 SXM, it runs at ~200 tok/s (2550 GB/s) for 256-token outputs.

## Model files

calm uses [🤗 Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model hyperparameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

[^1]: CUDA runtime and compiler is used for GPU acceleration, but no CUDA or C libraries are used. Python conversion scripts use safetensors and torch, see `tools/requirements.txt`.
[^2]: Linux is the main supported OS at the moment; calm also works on macOS (on CPU) but does not support Metal.
[^3]: Based on testing a specific Gigabyte GeForce RTX 4090 where both individual kernels from this repository and cuBLAS peak at about ~955 GB/s.
