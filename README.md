# 😌 calm

This is an experimental implementation of language model inference, aiming to get maximum single-GPU single-batch hardware utilization for LLM architectures with a minimal implementation. It is heavily work in progress.

If you need support for a wide range of models, computing devices or quantization methods, you're probably looking for [llama.cpp](https://github.com/ggerganov/llama.cpp) or [🤗 Transformers](https://github.com/huggingface/transformers).

This code is partially based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Model format

calm uses [Safetensors](https://huggingface.co/docs/safetensors/index) to store model files. Note that the models require conversion (see below), because calm stores model parameters in .safetensors metadata and may expect a particular set of tensor names or weight order within tensors that is not always compatible with the source. Tokenizer data is stored as tensors inside the model file as well.

## Running Llama-2-7B

```
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
python tools/convert.py llama-2-7b-chat.calm Llama-2-7b-chat-hf/
make && ./build/run ./llama-2-7b-chat.calm -i "Q.E." -n 17 -t 0
```

## Running Mistral-7B

```
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
python tools/convert.py mistral-7b-instruct.calm Mistral-7B-Instruct-v0.2/
make && ./build/run ./mistral-7b-instruct.calm -i "Q: 2+2?" -t 0 -n 12
```
