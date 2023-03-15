# One Thousand Tokens Per Second

The goal of this project is to research different ways of speeding up LLM inference, and then packaging up the best ideas into a library of methods people can use for their own models, as well as provide optimized models that people can use directly.

# Current Work
- Baseline benchmarks
- Evaluation framework
- Kernl vs TensorRT
- SparseGPT

# Benchmarks
| Model         | Engine | TPS   |
|---------------|--------|-------|
| ChatGPT Turbo | OpenAI | 15-20 |



Colab A100-40GB GPU, BS=1, sequence length=128, use_cache=True, tf32=True
| Model         | Engine | Dtype     | TPS   | Perplexity | Memory(GB) |
|---------------|--------|-----------|-------|------------|------------|
| Llama 7B      | HF     | INT8      | 5.7   | Base model | 8.7        |
| Llama 7B      | HF     | FP16      | 17.7  | Base model | 14.2       |
| Llama 7B      | HF     | BF16      | 17.5  | Base model | 14.2       |
| GPT-J-6B      | HF     | INT8      | 5.1   | Base model | 7.9        |
| GPT-J-6B      | HF     | FP16      | 17.4  | Base model | 13.1       |
| GPT-J-6B      | HF     | BF16      | 17.2  | Base model | 13.1       |

Colab A100-40GB GPU, BS=1, sequence length=512, use_cache=True, tf32=True
| Model         | Engine | Dtype     | TPS   | Perplexity | Memory(GB) |
|---------------|--------|-----------|-------|------------|------------|
| Llama 7B      | HF     | INT8      | 4.9   | Base model | 8.8        |
| Llama 7B      | HF     | FP16      | 16.1  | Base model | 14.7       |
| Llama 7B      | HF     | BP16      | 16.2  | Base model | 14.7       |
| GPT-J-6B      | HF     | INT8      | 5.2   | Base model | 8.5        |
| GPT-J-6B      | HF     | FP16      | 16.6  | Base model | 13.1       |
| GPT-J-6B      | HF     | BF16      | 16.3  | Base model | 13.1       |

## Learnings

- INT8 uses a bit more than half the amount of memory as the 16 bit variants, at the cost of a 3-4x decrease in inference speed.
- Llama is about the same speed as GPT-J, despite having 1B more params
- We need a ~60x speedup to reach 1k TPS

# Where to get speed gains

- Stability has a 4.2B param LLM that should be released soon, which should hopefully give a ~2x speedup
- Multi query attention can increase throughput by [reducing memory usage during inference](https://discordapp.com/channels/729741769192767510/1010280570921697372/1083560241410617435), but does not actually increase decoding speed
- [Kernl](https://github.com/ELS-RD/kernl/tree/main) or [TensorRT](https://github.com/NVIDIA/TensorRT) should give a ~4x speedup
- [SparseGPT](https://arxiv.org/abs/2301.00774) can prune half of the weights, but the model would most likely need retraining
- The [Terraformer](https://arxiv.org/pdf/2111.12763.pdf) architecture also uses sparsity that can bring ~5x speed increases, but is only meant for CPU, and to implement we would need to do model surgery to add in the controllers, so retraining would be needed.
