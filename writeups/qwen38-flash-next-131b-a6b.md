# Cutting the 51B n-gram table out of Qwen3.8-Flash-Next

*Project log, August 2026. Constraints: final GGUF under 65 GB, no training budget at all. Total cloud spend: $8, of which $6.50 was pods that never did anything. The campaign ran in one day.*

**Model weights:** [huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF](https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF)

**Reproduce it:** `surgery_qwen38.py` in the repo root, run against [unsloth's UD-Q3_K_XL shards](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). No GPU, no training, ~45 minutes on an NVMe laptop:

```
python surgery_qwen38.py Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf out.gguf --ple-keep 1
```

## TL;DR

Qwen3.8-Flash-Next spends 51B of its 176B parameters on a hash-embedding table of bigrams and trigrams. I removed 87.5% of that table — and nothing else — and the model still scores at parity with the full 90 GB quant on tool-calling, GSM8K, and MMLU, at **64.8 GB**. The only measurable cost is wikitext perplexity (2.40 → 4.66): the table turns out to be a surface-level autocomplete organ, not a reasoning one.

I also tried the classic prune, removing layers, the same cut that worked on Qwen3.6 after healing. On this architecture it collapses without healing, and healing was out of budget by design. The table cut is the whole result.

## Why quantization alone couldn't get there

unsloth's quant ladder for this model flattens out: Q3_K_XL is 90 GB, Q2_K_XL is 78.9 GB, and even the 1-bit build is 72.5 GB. The floor exists because the n-gram table is a third of the model and embedding-grade tensors resist low-bit quantization. Every gigabyte below ~72 has to come from removing parameters, not shrinking them.

## The table, and why it was cuttable

The table is 16 independent hash heads — 8 for bigrams, 8 for trigrams — each ~20M rows of 160 dims, each indexed by `hash % (its own prime)`, concatenated and injected at layer 1. Three properties made the cut nearly free to attempt:

1. **Multi-head hashing is redundant by design.** Each n-gram is looked up 8 independent ways per gram type, a Bloom-filter-style margin against collisions. Dropping heads degrades resolution gradually; there's no cliff to fall off.
2. **The layout is pure metadata.** llama.cpp reads head offsets and vocab sizes from GGUF keys and the row count from the tensor itself. Setting a dropped head's vocab to 1 and pointing it at a shared all-zero row silences it with no code patch — an all-zero IQ4_NL block decodes to exact zeros.
3. **The surgery never touches quantized data.** Kept rows are byte-for-byte copies out of unsloth's file, so their imatrix calibration survives untouched. There is nothing to retrain because nothing trained was changed.

The one subtlety: the primes killed my first idea (folding the table by an integer divisor is exact modular arithmetic only when the divisor divides the modulus — a prime has no divisors). Head-dropping replaced it and turned out cleaner anyway.

## Results

All rows same llama.cpp build (b10673, CUDA, A100), temp 0, same day. Suites as in the Qwen3.6 project: 40 agentic tool cases, GSM8K-15, MMLU-30, wikitext-2 perplexity at 32×512.

| model | GB | tools | GSM8K | MMLU | ppl |
|---|---:|---:|---:|---:|---:|
| base UD-Q3_K_XL | 90 | .900 | 1.000 | .833 | 2.40 |
| **keep 2/16 heads (shipped)** | **64.8** | 1.000 | 1.000 | .833 | 4.66 |
| table deleted entirely | 61.2 | .975 | 1.000 | .767 | 4.80 |
| ½ table + 12 layers cut | 60.5 | .650 | .667 | .600 | 13.2 |
| ¼ table + 12 layers cut | 53.3 | .825 | .333 | .733 | 13.7 |

Two honesty notes. Tool accuracy has a ±3-case band between CPU and CUDA backends at temp 0 (a CPU control scored the base at .975), so "1.000 vs .900" means *parity*, not a win. And these are small suites; GSM8K and MMLU deltas of one question are noise. Perplexity is the reliable continuous signal, and it replicated across environments to three decimals.

The shipped config keeps 1 head of 8 per gram type — the most table that fits under 65 GB. It was never worse than full deletion on any metric, so the 3.6 GB it costs over the fully-gutted build is cheap insurance.

## The dead end: layer cuts on this architecture

On Qwen3.6, removing 1-in-4 recurrent layers left an unhealed model that still called tools at .70 and did GSM8K at .80 — damaged but clearly recoverable, and $53 of LoRA healing recovered it. I ran the equivalent cut here (48 → 36 layers, same uniform-interval discipline, preserving the table-injection layer) expecting the same shape of damage.

It wasn't the same shape. Both cut variants cratered: GSM8K to .33–.67, perplexity to 13+, and the damage didn't even scale sensibly with the size of the accompanying table cut. My working suspicion is the 4-branch hyper-connection residual stream — layers in this architecture are more entangled with their neighbors than a plain residual stack, so removing one hurts more. I didn't verify the mechanism; what's verified is the outcome. **Unhealed layer pruning is dead on qwen4exp**, and going below ~55 GB would need a healing budget several times what the 3.6 heal cost, chasing a gap three times as large.

## What I took away from it

1. **Look for the parameter-scaling organ before cutting compute.** New architectures increasingly park capacity in cheap-to-read side structures (this table; per-layer embeddings elsewhere). Those are prunable in ways transformer weights are not: graceful degradation, no retraining, no interaction with routing or attention.
2. **Perplexity and task accuracy can decouple completely.** A 2× perplexity hit with zero task damage would have looked like a disaster if I'd used ppl as the gate. The table predicts surface text; it doesn't reason. Measure both, trust neither alone.
3. **Pruning lessons don't transfer across architectures.** The exact cut that was "damaged but healable" on Qwen3.6 is fatal one generation later. Re-run the crude prune-and-bench experiment on every new architecture before believing anything.
4. **Byte-verbatim surgery beats requantization.** Copying quantized blocks preserves someone else's careful imatrix work for free. The only tensors I wrote were the ones I changed.
5. **Same-day, same-binary controls, still.** The tools ±3-case CPU/CUDA band would have read as a real effect against last week's baseline numbers.

## License

Code in this repo is Apache-2.0 — see [LICENSE](../LICENSE). Model weights are distributed on Hugging Face under qwen-community-1.0, inherited from the base model.
