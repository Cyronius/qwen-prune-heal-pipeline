# qwen-prune-heal-pipeline

Shrinking Qwen MoE models to run well on shared-memory hardware (CPU / iGPU boxes, no discrete GPU required). Each campaign takes a released checkpoint, finds what can be cut, benchmarks the damage honestly, and ships the result on Hugging Face. Failures are documented as carefully as the wins — they're most of the useful information.

## Shipped models

| model | from | size | method | writeup |
|---|---|---:|---|---|
| [Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B) | Qwen3.6-35B-A3B (22.9 GB Q4) | 16.5 / 12 GB | layer prune + $53 LoRA heal + MTP speculative decode | [the prune-and-heal campaign](writeups/qwen36-27b-a2.8b.md) |
| [Qwen3.8-Flash-Next-131B-A6B](https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF) | Qwen3.8-Flash-Next (90 GB Q3) | 64.8 GB | n-gram hash-table prune, zero training | [the no-heal campaign](writeups/qwen38-flash-next-131b-a6b.md) |

Two campaigns, nearly opposite conclusions, one generation apart: on Qwen3.6, layer pruning was survivable and $53 of healing recovered it; on Qwen3.8's architecture the same cut is fatal — but 87.5% of its 51-billion-parameter n-gram table came out for free, with task scores at parity with the full model.

## Repo structure

| path | what's there |
|---|---|
| `prune_qwen35.py` | Qwen3.5/3.6 layer-cut / expert-width-cut script (safetensors-level) |
| `surgery_qwen38.py` | Qwen3.8-Flash-Next GGUF→GGUF surgery: n-gram head dropping, layer cuts |
| `heal/` | LoRA healing pipeline: teacher generation, training, merging, CPU smoke test |
| `bench/` | benchmark harness (tools / GSM8K / MMLU / perplexity), pod orchestration, result logs |
| `patches/` | llama.cpp tokenizer-hash patch (only needed for pre-Aug-2026 checkouts) |
| `writeups/` | the full campaign write-ups behind every number |
| `MODEL_CARD*.md` | the Hugging Face model cards |

Model weights, adapter checkpoints, and quantized builds aren't in this repo — see the Hugging Face links above.

## License

Code is Apache-2.0 — see [LICENSE](LICENSE). Model weights are distributed separately on Hugging Face under their own licenses (Apache-2.0 for the 3.6 derivative, qwen-community-1.0 for the 3.8 derivative).
