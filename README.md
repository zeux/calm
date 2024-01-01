# 😌 calm

This is an implementation of language model inference, aiming to get maximum single-GPU single-batch hardware utilization for LLM architectures with a minimal implementation and minimal dependencies[^1].

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

calm currently expects models to follow Llama-like architecture (RMSNorm normalization, SiLU activation, sequential attention mixing and FFN, RoPE). It has been tested on following models:

- Llama2 7B (meta-llama/Llama-2-7b-chat-hf)
- Mistral 7B (mistralai/Mistral-7B-Instruct-v0.2)
- SOLAR 10.7B (upstage/SOLAR-10.7B-Instruct-v1.0)
- Yi 34B (01-ai/Yi-34B-Chat)
- TinyLlama 1.1B (TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## Supported formats

Model weights support `fp16` and `fp8` formats; the weight type is specified at conversion time via `--dtype` argument to `convert.py`.

`fp16` corresponds to 16-bit floating point (e5m10) and is the default option; note that some models store weights in bf16 which will be automatically converted.

`fp8` corresponds to 8-bit floating point (e5m2). Using `fp8` carries a ~0.5% perplexity penalty at almost double the inference speed and half the model size. e4m3 variant of `fp8` would result in a much smaller perplexity penalty (~0.1%) with basic tensor scaling, but it's currently not used because of performance issues wrt floating-point conversion.

KV cache is using `fp16`.

## Performance

As of December 2023, with Mistral 7B model and `fp16` weights, `calm` reaches ~63.5 tok/s (921 GB/s) on short sequences and ~60 tok/s (904 GB/s) at the end of the 4096 token context, when using NVidia GeForce RTX 4090. When using `fp8` weights on the same hardware and model, the performance is ~119 tok/s (863 GB/s) on short sequences and ~108 tok/s (840 GB/s) at the end of the context.

Currently prompts are processed serially, one token at a time; in the future, prompt processing will need to be parallelized to avoid the bandwidth bottleneck.

Currently weights support `fp16` and `fp8` formats; in the future, 4-bit quantization is planned. This will allow running inference at higher tok/s, however the main metric is bandwidth utilization and the goal is to keep it as close to peak as possible at all supported weight formats.

RTX 4090 has a peak bandwidth of ~1008 GB/s, however it's unclear if a peak higher than ~950 GB/s is attainable in practice[^3]. The code has not been heavily tuned for datacenter-grade hardware (A100/H100) or earlier NVidia architectures yet.

## Model files

calm uses [🤗 Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model hyperparameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

[^1]: CUDA runtime and compiler is used for GPU acceleration, but no other CUDA libraries are used; [jsmn](https://github.com/zserge/jsmn) is used for JSON parsing. Python conversion scripts use safetensors and torch, see `tools/requirements.txt`.
[^2]: Linux is the only supported OS at the moment.
[^3]: Based on testing a specific Gigabyte GeForce RTX 4090 where both individual kernels from this repository and cuBLAS peak at about ~955 GB/s.
