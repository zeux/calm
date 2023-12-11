# calm - (CUDA) accelerated language models

This is an experimental implementation of language model inference, aiming to get maximum single-GPU hardware utilization for LLM architectures with a minimal implementation. It is heavily work in progress.

If you need support for a wide range of models, computing devices or quantization methods, you're probably looking for [llama.cpp](https://github.com/ggerganov/llama.cpp) or [🤗 Transformers](https://github.com/huggingface/transformers).

This code is partially based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).