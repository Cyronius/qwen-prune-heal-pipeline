# Reduced Top-k Routing + Heal Plan (possibility, not started)

**Target model:** Qwen3.6-35B-A3B (MoE, 256 routed experts + 1 shared expert, default `k=8`)
**Goal:** Run at `k=4` routed experts per token without the quality collapse seen when simply overriding `k`.
**Status:** Idea only. Nothing here has been run yet.

---

## Problem

Dropping from 8 to 4 routed experts per token on the full 35B badly damages the model. On the healed 27B (`heal-artifacts/topk-bench.log`) the same change hurt, but didn't break it:

| k | tools | GSM8K | MMLU |
|---|---:|---:|---:|
| 8 | .975 | .867 | .767 |
| 6 | .925 | .867 | .767 |
| 4 | .900 | .733 | .700 |

So "destroyed" on the 35B may partly be a setup problem, not only lost capacity.

## Step 0 — rule out a scaling bug (free)

The router does a softmax over all experts, takes the top k, and (with `norm_topk_prob`) rescales the kept weights to sum to 1. If that rescale is skipped, the top 4 may carry only ~60% of the weight. That shrinks every MoE block's output, and the error compounds over 40 layers.

- **Hugging Face:** set `num_experts_per_tok=4`, confirm `norm_topk_prob` is true.
- **llama.cpp:** `--override-kv qwen35moe.expert_used_count=int:4` (as in `heal-artifacts/mtp_probe.py`), and confirm the GGUF has expert-weight normalization on.

Re-bench. If quality is now close to the 27B numbers above, the remaining steps may be optional.

## Step 1 — keep the router, lose less (no training)

The router doesn't need to change. Top-4 is just the first four of its top-8, so its ranking is still valid. What's lost is the output of experts 5–8. Options:

- **Per-layer k:** measure each layer's sensitivity to k=4. Keep k=8 in the few worst layers (often first and last), k=4 elsewhere.
- **Adaptive k:** keep adding experts until their summed weight passes a threshold (e.g. 0.8). Easy tokens use 3–4, hard ones up to 8. Needs runtime/kernel support.
- **Settle for k=6:** in the table above it matched k=8 on GSM8K and MMLU.

## Step 2 — heal at k=4

Reuses the existing `heal/` pipeline.

1. **Teacher:** the unmodified k=8 model. Run `heal/gen_teacher.py` to store top-64 log-probs.
2. **Student:** same weights, `num_experts_per_tok=4`.
3. **Train:** `heal/train_heal.py` in `kd` mode (KL against the teacher). KD fits a routing change better than plain SFT.
4. **What trains:** the existing LoRA targets in `heal/common.py` (attention + shared expert). The shared expert runs on every token, so it can absorb the average of what the dropped experts used to add.
5. **Router (optional):** unfreeze `mlp.gate.weight` directly (`requires_grad=True`, low LR). PEFT can't reach it because it's a raw `nn.Parameter`. This teaches the router to pick the best 4 rather than the old top-4. It's small (hidden × 256 per layer), so it's cheap.

Code changes needed when this starts: an `--experts-per-tok` flag on the student load, and a `--train-router` flag in `train_heal.py`.

**Expectation:** based on the layer-prune heal (~$53, a few hundred steps), recover most of the gap but not all. k=4 really does halve routed compute.

## Step 3 — confirm the speedup is real before paying for a heal

- The writeup (`writeups/qwen36-27b-a2.8b.md`, "Where the real speedup came from") notes the decode speedup from lower k **never reproduced reliably** on the target hardware.
- The MTP draft head was trained against k=8 routing; k=4 may lower its acceptance rate.

Benchmark k=4 vs k=8 decode several times, with and without MTP. If k=4 isn't clearly faster, k=8 + MTP stays the better ship, and this plan isn't worth running.
