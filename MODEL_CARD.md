---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
tags:
- moe
- pruning
- quantization
- gguf
- tool-calling
- llama.cpp
pipeline_tag: text-generation
---

# Qwen3.6-27B-A2.8B

Pruned, LoRA-healed, and quantized derivative of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), built for fast decode on shared-memory hardware (CPU/iGPU, no discrete GPU required). 10 of 40 language-model layers were removed (1 of every 3 gated-DeltaNet layers), the result was LoRA-healed to recover quality, then quantized to GGUF.

| | Qwen3.6-35B-A3B (base) | Qwen3.6-27B-A2.8B (this model) |
|---|---:|---:|
| total params (LM core) | 34.7B | 26.2B |
| active params / token | 3.45B | 2.83B |
| layers | 40 | 30 |

## Files

Two GGUF builds, both with the multi-token-prediction (MTP) draft head retained for speculative decoding:

| file | size | quant | notes |
|---|---:|---|---|
| `qwen36-27b-a2.8b-mtp-Q4KM.gguf` | 16.49 GB | Q4_K_M, imatrix | balanced — best overall accuracy |
| `qwen36-27b-a2.8b-mtp-iq3exp-q4head.gguf` | 12.01 GB | routed experts → IQ3_S, `lm_head` → Q4_K, rest unchanged | speed-demon — smaller and faster, some accuracy cost |

Both share the same imatrix and the same attention / GDN / MTP tensors; the speed-demon build only pushes the routed-expert and output-head tensors to a smaller quant.

## Running it

llama.cpp / llama-server, with MTP speculative decoding enabled:

```
llama-server -m qwen36-27b-a2.8b-mtp-Q4KM.gguf --spec-type draft-mtp --spec-draft-n-max 2
```

Needs a llama.cpp build with MTP support. Default routed-expert count (`k=8`) is recommended — dropping it did not give a reproducible speed gain in testing (see Limitations).

## Benchmarks

Tool-calling: agentic tool-use cases scored on whether the model calls the right tool with the right arguments, or correctly declines. GSM8K / MMLU: standard accuracy. Perplexity: wikitext-2 test set, 32 chunks, context 512, llama.cpp `perplexity` tool. Decode: ordinary generation, tokens/sec. Tool decode: generation while producing tool-call output.

| model | size | tools | GSM8K | MMLU | ppl | decode | tool decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| **balanced (Q4_K_M)** | 16.49 GB | **.95** | .80 | .733 | 8.01 | 36.8 | 40.7 |
| **speed-demon (IQ3_S experts)** | **12.01 GB** | .90 | .733 | .767 | 8.29 | **40.7** | **44.1** |
| base Q4_K_M (Qwen3.6-35B-A3B, for reference) | 21.17 GB | .90 | .933 | .767 | 5.50 | 27.8 | 27.9 |

Decode tok/s varied between repeat runs; treat those two columns as directional, not exact. Full result logs are in the GitHub repo's `bench/results/`.

## Limitations

- LoRA healing covered roughly half an epoch (18K prompts / 16M tokens). GSM8K is still behind the base model, and perplexity recovered most but not all of the way.
- The speed-demon build trades tool accuracy and GSM8K for size and speed.
- MTP speculative decoding is not bit-exact vs. plain decoding: a 40-case check scored 37/40 vs. 39/40 for plain decoding. Functionally equivalent, not guaranteed identical.
- Lowering routed-expert count (`k`) below 8 did not give a reproducible speed benefit in testing — not recommended.
- Evaluated on a small in-house benchmark (7 tool schemas, GSM8K/MMLU subsets); results may not generalize to other workloads.

## How this model was made

Full write-up, code, and result logs: [github.com/Cyronius/qwen-prune-heal-pipeline](https://github.com/Cyronius/qwen-prune-heal-pipeline)

Short version: 1-in-3 gated-DeltaNet layers were structurally removed from the 40-layer base, then a rank-32 LoRA (~33M trainable params, ~$53 of H200 time) was trained on a mix of general text, tool-use, and GSM8K-style reasoning data to heal the cut, then the result was quantized to GGUF with a calibrated imatrix.

## License

Apache-2.0, inherited from the base model, [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
