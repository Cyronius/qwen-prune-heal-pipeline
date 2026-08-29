---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- moe
- pruning
- quantization
- gguf
- hash-embedding
- llama.cpp
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next-131B-A6B (n-gram table pruned, GGUF)

A 64.8 GB GGUF of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) that scores at parity with the full 90 GB quant on tool-calling, GSM8K, and MMLU — 28% smaller, with **zero training and zero transformer surgery**.

Qwen3.8-Flash-Next carries a 51B-parameter n-gram hash-embedding table (~29% of total weights): 16 independent hash heads, 8 per n-gram type (bigrams / trigrams), injected at layer 1. This model keeps **1 head of 8 per n-gram type** and drops the other 14, exploiting the built-in redundancy of multi-head hashing (each n-gram is looked up several independent ways; a Bloom-filter-style safety margin). Every other tensor is copied **byte-for-byte** from [unsloth's UD-Q3_K_XL dynamic quant](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), so unsloth's imatrix calibration is preserved exactly.

| | Qwen3.8-Flash-Next (base) | this model |
|---|---:|---:|
| total params | ~176B | ~131B |
| active / token | 6B | 6B |
| n-gram table | 51.2B (16 heads) | 6.4B (2 heads) |
| layers / experts | 48 / 512 | 48 / 512 (untouched) |
| UD-Q3_K_XL GGUF size | 90 GB | **64.8 GB** |

## Benchmarks

All rows measured on the same llama.cpp build (b10673, CUDA, A100), temp 0, same day. Tool-calling: 40 agentic cases scored on calling the right tool with the right arguments or correctly declining (schemas disjoint from anything the surgery could see — there is no training step). GSM8K-15 / MMLU-30 subsets, no-think mode. Perplexity: wikitext-2 test, 32×512-token chunks.

| model | size | tools | GSM8K | MMLU | ppl |
|---|---:|---:|---:|---:|---:|
| base UD-Q3_K_XL | 90 GB | .900 | 1.000 | .833 | 2.40 |
| **this model (2/16 heads)** | **64.8 GB** | 1.000 | 1.000 | .833 | 4.66 |
| table fully deleted (0/16) | 61.2 GB | .975 | 1.000 | .767 | 4.80 |

Tool accuracy has a ±3-case run-to-run band across hardware backends at temp 0 (a CPU control run scored the base at .975) — read "1.000 vs .900" as *parity with base*, not as beating it. The one real cost is perplexity: the n-gram table turns out to be a surface-level next-token predictor. Removing 87.5% of it doubles wikitext perplexity while leaving reasoning and tool use at baseline.

For calibration: both cheaper cuts were also tested and rejected. Removing transformer layers (the classic prune) collapsed this architecture without healing — GSM8K fell to 0.33–0.67 and perplexity hit 13+ at 53–60 GB. The n-gram cut is the only free lunch here.

## Running it

Needs llama.cpp from **2026-08-27 or newer** (`qwen4exp` support; b10673 tested). Vendor-recommended sampling: instruct `temp 0.7, top_p 0.8, presence_penalty 1.5`; thinking `temp 1.0, top_p 0.95`.

```
# ~66 GB+ VRAM: fully offloaded
llama-server -m qwen38-keep1-Q3KXL.gguf -ngl 99 -c 8192

# smaller GPUs: keep the (sparse-gather) n-gram table in system RAM
llama-server -m qwen38-keep1-Q3KXL.gguf -ngl 99 -c 8192 \
  --override-tensor "per_layer_token_embd\.weight=CPU"

# CPU-only boxes with ~70 GB free RAM work too (slow but correct)
```

Text-only GGUF (vision tensors ship separately as unsloth's mmproj; the vision path is untested with this surgery). MTP speculative decoding is not available — upstream llama.cpp does not export or run the MTP head for this architecture yet.

## Limitations

- Wikitext perplexity 4.66 vs 2.40 for the base: prose is measurably less "polished-autocomplete" even though task performance holds. If your workload is verbatim-recall-heavy (quotes, boilerplate reproduction), the missing table may show up.
- Evaluated on a small in-house benchmark (40 tool cases, GSM8K-15, MMLU-30) at ≤8K context. Long-context behavior (QSA sparse attention, 256K native) untested.
- The kept heads were chosen positionally (first of each 8), not by importance ranking. A calibrated head choice might do slightly better; nobody has measured.

## How this model was made

Full write-up, surgery script, and raw result logs: [github.com/Cyronius/qwen-prune-heal-pipeline](https://github.com/Cyronius/qwen-prune-heal-pipeline) (`surgery_qwen38.py`).

Short version: the GGUF's per-head hash-table layout is entirely metadata-driven (`ple.head_offsets`, `ple.head_vocab_sizes`), and llama.cpp reads the table's row count back from the tensor itself. Dropped heads get `vocab_size = 1` pointing at a shared all-zero row (an all-zero IQ4_NL block decodes to exact zeros), so they contribute nothing and cost one row of storage — no llama.cpp patch, no requantization, no training. The whole build is a streaming byte copy: ~45 minutes on an NVMe laptop, and the entire experimental campaign that selected this configuration cost $1.50 of A100 time.

## License

qwen-community-1.0, inherited from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Quantized weights derived from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — thanks to the unsloth team for the calibrated dynamic quant this build inherits.
