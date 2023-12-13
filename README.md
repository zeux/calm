# calm

This is an experimental implementation of language model inference, aiming to get maximum single-GPU hardware utilization for LLM architectures with a minimal implementation. It is heavily work in progress (for example, it doesn't run on GPU yet!).

If you need support for a wide range of models, computing devices or quantization methods, you're probably looking for [llama.cpp](https://github.com/ggerganov/llama.cpp) or [🤗 Transformers](https://github.com/huggingface/transformers).

This code is partially based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

## Running Llama-2-7B

```
git clone https://huggingface.co/meta-llama/llama-2-7b-chat-hf
python tools/convert.py llama-2-7b-chat.safetensors --models llama-2-7b-chat-hf/pytorch_model-00001-of-00002.bin llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin
python tools/tokenizer.py llama-tokenizer.bin --model llama-2-7b-chat-hf/tokenizer.model
make && ./build/run ./llama-2-7b-chat.safetensors -z llama-tokenizer.bin -i "Q.E." -n 17 -t 0
```

## Running Mistral-7B

```
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
python tools/convert.py mistral-7b-instruct.safetensors --models Mistral-7B-Instruct-v0.1/pytorch_model-00001-of-00002.bin Mistral-7B-Instruct-v0.1/pytorch_model-00002-of-00002.bin
python tools/tokenizer.py mistral-tokenizer.bin --model Mistral-7B-Instruct-v0.1/tokenizer.model
make && ./build/run ./mistral-7b-instruct.safetensors -z mistral-tokenizer.bin -i "Q: 2+2?" -t 0 -n 12
```
