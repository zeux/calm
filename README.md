# 😌 calm

This is an experimental implementation of language model inference, aiming to get maximum single-GPU single-batch hardware utilization for LLM architectures with a minimal implementation and no dependencies[^1]. It is heavily work in progress.

If you need support for a wide range of models, computing devices or quantization methods, you're probably looking for [llama.cpp](https://github.com/ggerganov/llama.cpp) or [🤗 Transformers](https://github.com/huggingface/transformers). If you need to run inference for multiple batches, you're probably looking for [vLLM](https://github.com/vllm-project/vllm).

Parts of this code are based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Running

To build and run `calm`, you need to download and convert a model, build the code using `make`[^2] and run it:

```sh
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python tools/convert.py mistral-7b-instruct.calm Mistral-7B-Instruct-v0.2/
make && ./build/run mistral-7b-instruct.calm -i "Q: What is the meaning of life?" -t 0
```

Before running Python you may want to install the dependencies via `pip -r tools/requirements.txt`. When using git, git-lfs is required and the download size may be larger than necessary; you can use `tools/download.py` instead:

```sh
python tools/download.py Mistral-7B-Instruct-v0.2/ mistralai/Mistral-7B-Instruct-v0.2
```

## Supported models

calm currently expects models to follow LLama-like architecture (RMSNorm normalization, SiLU activation, sequential attention mixing and FFN). It has been tested on following models:

- Llama2 7B (meta-llama/Llama-2-7b-chat-hf)
- Mistral 7B (mistralai/Mistral-7B-Instruct-v0.2)
- SOLAR 10.7B (upstage/SOLAR-10.7B-Instruct-v1.0)

## Supported formats

Model weights are currently using `float16`. KV cache activations are currently using `float16`. Support for smaller formats is planned.

## Model files

calm uses [🤗 Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model hyperparameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

[^1]: CUDA runtime and compiler is used for GPU acceleration, but no other CUDA libraries are used; [jsmn](https://github.com/zserge/jsmn) is used for JSON parsing. Python conversion scripts use safetensors and torch, see `tools/requirements.txt`.
[^2]: Linux is the only supported OS at the moment.
