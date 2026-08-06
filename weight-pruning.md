# 4:2 Structured Pruning + Q4 Quantization Plan

**Target model:** Qwen3.5-35B-A3B (MoE, ~3B active params/token)
**Target hardware:** AMD 890M iGPU (RDNA 3.5, Vulkan/llama.cpp path) — no native 2:4 sparse tensor core support (that arrives with RDNA4)
**Goal:** Custom structured sparsity format + kernel to push past Q4's compression ceiling, without relying on hardware sparse acceleration.

---

## Core idea

Standard N:M sparsity (e.g. NVIDIA/AMD's native 2:4) requires dedicated decoder hardware the 890M doesn't have. Instead, do the equivalent bucketing/gather work **once, at model-conversion time**, producing a static custom format that any Vulkan compute shader can consume as ordinary dense sub-matmuls. No sparse-specific silicon required — the "specialness" is baked into the file layout, not the runtime.

## Pattern scheme

- Group every row's weights into consecutive chunks of 4.
- Within each chunk, keep the 2 highest-importance weights, drop the other 2 (4:2 / 2:4 sparsity — 50% weight removal per group).
- There are exactly C(4,2) = 6 possible survivor patterns: `1100, 1010, 1001, 0110, 0101, 0011`.
- At conversion time, bucket all `(neuron, chunk)` pairs by which of the 6 patterns they landed in.
- Within each pattern bucket, gather the surviving weights + their input-column indices into a **contiguous dense sub-matrix**. Six dense sub-matrices per pruned layer, each pattern-homogeneous.
- Pad bucket sizes to a clean multiple (e.g. 32/64, matching AMD wavefront size) for SIMD alignment — accept a small amount of importance-ranking suboptimality in exchange for balanced, hardware-friendly bucket shapes.

## Pipeline order (matters)

1. **Prune first, at higher precision** (bf16/fp16), before any quantization. Importance scoring on unquantized weights gives cleaner signal.
2. **Importance metric:** don't use raw `|weight|` magnitude alone. Prefer Wanda-style scoring (`|weight| × input activation magnitude`, from a small calibration set) — consistently outperforms magnitude-only pruning in the literature and is cheap to compute.
3. **Post-prune reconstruction step:** after zeroing the losing weights, adjust the surviving weights per layer to compensate (SparseGPT-style least-squares reconstruction against calibration data). Skipping this is the most likely place to lose more quality than the plan predicts.
4. **Quantize survivors to Q4** (dynamic, e.g. Unsloth Dynamic-style per-layer bit allocation) — quantize *after* pruning, not before.
5. **Bucket/gather + pad**, write out the 6-sub-matrix custom format per pruned layer, with a precomputed offset/dispatch table (no dynamic indexing needed at inference time).

## Why 4:2 and not more aggressive (4:1, etc.)

- Pruning degrades accuracy in a flat-then-cliff curve, not linearly. ~50% sparsity is generally recoverable with light fine-tuning; 70%+ typically requires substantial retraining and sits past where most published structured-pruning results hold up.
- Unlike quantization (which spreads small, largely-cancelling error across every weight via averaging over the dot product), pruning deletes specific values entirely — no cancellation effect bails you out if an important weight was cut.
- MoE experts see less training signal per expert than a comparably-sized dense model (routing splits tokens across experts) — likely *less* redundancy to spare than dense layers, so err conservative on prune ratio for this architecture specifically.

## Bit budget (approximate, per 4-weight group, ignoring per-block scale/min overhead real GGUF formats add)

| Scheme | Bits/group | Bits/weight |
|---|---|---|
| Dense Q4 | 16 | 4.0 |
| Dense Q3 | 12 | 3.0 |
| 4:2 prune + Q4 survivors | 2×4 + ~3 (pattern idx) = 11 | ~2.75 |
| 4:2 prune + Q3 survivors | 2×3 + ~3 = 9 | ~2.25 (not recommended — compounds two lossy steps on the same weights, see below) |

**Decision: prune to 50% (4:2), quantize survivors to Q4 (dynamic).** Beats plain Q3 on size while likely beating it on accuracy — full 4-bit fidelity preserved on weights judged most important, rather than uniform degradation across all weights. Avoids stacking two aggressive lossy steps (pruning + Q3) on the same values with no full-precision step left to absorb error.

## Required engineering

1. **Conversion pipeline** (one-time, offline): prune → reconstruct → quantize → bucket/gather → write custom format + offset table.
2. **Custom Vulkan compute shader**: reads the 6 dense sub-matrices + pattern/offset table per pruned layer, runs 6 ordinary dense GEMMs, sums partial outputs per neuron. No sparse decode logic — this is the whole point of doing the gather at conversion time instead of runtime.
3. **llama.cpp integration**: this format has no existing loader/kernel support; expect to hook in at the GGUF tensor-type level or maintain a fork.

## Validation before trusting this for daily use

- Compare perplexity / small eval suite: **4:2-prune+Q4** vs. **plain Q4_K_M** vs. **Unsloth Dynamic Q4**, all at matched file size.
- If 4:2-prune+Q4 doesn't clearly beat matched-size baselines, the added kernel complexity isn't worth carrying.
- MoE-specific behavior under pruning is untested territory for this model — treat early results as exploratory, not assumed-safe.

## Realistic performance expectations

- Bottleneck on 890M is memory bandwidth, not compute — the win here comes from fewer bytes moved per token, not from cutting FLOPs.
- Expect roughly 15–25% additional decode speedup over dense Q4 on pruned layers if the kernel and pruning both land cleanly — not 2x, and not "half the time." Real deployed N:M kernels on hardware *built* for this (RDNA4/Ampere+) top out around ~1.3x over dense; a hand-rolled shader without hardware decode support should be expected to land under that on the compute side, though bandwidth savings should transfer more directly.