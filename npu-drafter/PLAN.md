# NPU-native speculative drafter — implementation plan

*Status: plan, October 2026. Supersedes "NPU_Native_Drafter_Plan.docx". First target: ternary Bonsai 2 (Qwen3.8-27B). Goal: one drafter design, built for the XDNA2 NPU, that ships inside oflm for several target models, with a per-target fine-tune.*

## 0. Summary

We build a small speculative drafter that lives on the Strix XDNA2 NPU. Its hot weights and memory stay on the chip. The target model runs on the iGPU and checks the drafts. The NPU and iGPU are separate devices, so drafting runs **at the same time as** verification, not in between.

The design has five main parts:

1. **Fixed-size memory.** DeltaNet-style layers keep a per-head state inside the compute tiles. One optional full-attention layer keeps exact KV in the memory tiles.
2. **Trimmed vocab.** An 8k–16k core token set, plus dynamic slots filled with tokens from the current context.
3. **Learned memory gate.** As tokens age out of the recent window, a gate decides **keep / merge / dump** for each one.
4. **Overlap and pre-drafting.** The NPU drafts the next block, and likely branches, while the iGPU verifies the current one.
5. **One backbone, many targets.** A shared drafter core plus small per-target parts (input adapter, vocab map, head, gate), fine-tuned per target and shipped as a versioned package.

The primary metric is **accepted tokens/sec end to end, at a stated quality setting**. Lossless mode is the default and the reference. Lossy modes are an explicit, measured quality budget.

## 1. Decisions already made

These came out of the plan review. Change them only with data.

| # | Decision | Why |
|---|---|---|
| D1 | The vocab head is never full size. Use a trimmed core set (8k–16k) + context slots (256–1024). | A 150k+ head is ~10 MB even at 2 bits; total on-chip memory is ~6 MB. |
| D2 | Embedding lookup happens on the host. Only vectors cross to the NPU. | The table doesn't fit; the lookup is cheap on the host. |
| D3 | The target-side feature projection runs on the target device (iGPU) as part of its forward pass. | The multi-layer → d projection is millions of parameters. |
| D4 | The core memory is a fixed-size recurrent state (DeltaNet-style), one head per compute tile. | It fits SRAM at any context length, and maps cleanly to the matrix engines. |
| D5 | Long-term exact recall comes from at most 1–2 full-attention layers, with KV split across the 8 memory tiles by context chunk and merged with log-sum-exp. | Exact recall for names, identifiers and numbers, with balanced columns. |
| D6 | Weights live in the memory tiles and stream into all 32 compute tiles one stage at a time. No fixed spatial Draft/Refine split. | A spatial split idles half the array when stages run in sequence. |
| D7 | Refinement first uses **shared weights + a step embedding**. Separate Stage-2 weights only if that loses clearly. | It doubles capacity per stage. |
| D8 | Drafting overlaps verification. The NPU pre-drafts likely branches. | Verify time (tens of ms) dwarfs draft compute (µs to sub-ms). |
| D9 | Only **accepted** tokens age out of the recent window into long-term memory. | Memory-gate decisions never need rollback. |
| D10 | "Zero DRAM weight reads" is a measured trade-off, not a rule. | Its real value is avoiding bus contention with the target; that is measured, not assumed. |
| D11 | Precision is chosen by measurement: INT8 vs INT4 vs ternary, including unpack cost. | Ternary saves bytes, not MACs: there's no native ternary MAC path. |
| D12 | Lossless is the default. Relaxed acceptance is an opt-in quality budget, scored on `bench/`. | Same honesty rule as the prune campaigns. |

## 2. System architecture

```
                 host CPU (oflm)
   tokens ─┬─ embedding lookup (core vocab + context slots)
           │
           ▼                     shared DDR ring buffer
   ┌───────────────────────── XDNA2 NPU (persistent kernel) ─────────────────────────┐
   │ memory tiles (8 × 512 KB)                                                        │
   │   drafter weights (~2.5–3 MB)   long-term KV slots: keep + merge (~1–1.5 MB)     │
   │        │ stream per stage            │ split by column                           │
   │        ▼                             ▼                                           │
   │ compute tiles (32 × 64 KB)                                                       │
   │   recent window · DeltaNet state (1 head/tile) · GEMM working set                │
   │   loop: draft → refine × R (shared weights + step embedding)                     │
   │   out: N-token block (+ branch blocks), top-m q per position, confidence         │
   └──────────────────────────────────────────────────────────────────────────────────┘
           │ drafts                                  ▲ accepted k, correction token,
           ▼                                         │ projected features, top-k scores
   ┌──────────── iGPU: target model (e.g. Bonsai 2) ─┴───────────┐
   │ verify block/tree in one pass · state rollback ·            │
   │ feature adapter (target layers → d) appended to the forward │
   └─────────────────────────────────────────────────────────────┘
```

### 2.1 Per-cycle flow

1. The iGPU verifies block t and, in the same pass, emits: accepted length k, the correction token, projected features for the accepted positions, and the target's top-k scores.
2. Meanwhile, the NPU has drafted block t+1 for the most likely outcomes of step 1: full accept, plus a few "rejected at j, correction = drafter's 2nd/3rd choice" branches.
3. When the result arrives, the host picks the matching branch. If one matches, the next verify starts at once. If none matches, the NPU drafts from the true state, which is the slow path.
4. Accepted tokens move into the NPU's recent window. Tokens leaving that window pass through the memory gate.

### 2.2 Lossless sampling

- **Greedy:** accept while draft == target argmax. The correction is the argmax.
- **Temperature > 0:** the drafter samples from a **top-m truncated, renormalized** distribution q (for example m = 32). That q is exactly what was sampled from, so sending its m values is enough for the standard rule (accept with min(1, p/q); on reject, sample from norm(max(0, p − q))). No full-vocab q is ever needed. With refinement, q is the distribution of the **final** pass that chose the token.
- **Tree verification** (optional, phase 9) uses the same rule per path.

## 3. Hardware budget (planning figures; phase 1 replaces them)

| Resource | Planning value | Use |
|---|---|---|
| Compute tiles | 32 × 64 KB data memory (~2 MB) | state, recent window, GEMM working set |
| Memory tiles | 8 × 512 KB (~4 MB) | weights + long-term KV |
| Weight budget | ~2.5–3 MB | ~10M params at INT4/ternary, ~3M at INT8 |
| Trimmed head, 8k × 256 | 0.5 MB ternary / 1 MB INT4 / 2 MB INT8 | factorize (8k×64 + 64×256) if tight |
| DeltaNet state | 64×64 bf16 ≈ 8 KB per head | one head per tile |
| Long-term KV | ~128 B/token (1 head, dim 64, int8) | ~12K tokens in 1.5 MB, one layer |
| Draft compute | ~0.5 GOP per 16-token block × 4 loops | µs–sub-ms; not the bottleneck |
| Target verify | tens of ms per pass | the bottleneck; measured in phase 0 |

## 4. Multi-target design (oflm)

### 4.1 What is shared and what is per-target

| Part | Shared across targets | Per target |
|---|---|---|
| NPU kernels / compiled binary | yes, one per **size class** | — |
| Drafter core (DeltaNet + attention + refine) | pretrained once per tokenizer family | fine-tuned |
| Input adapter (target layers → d) | — | yes; runs on the target device |
| Core vocab set + context slot logic | per tokenizer family | coverage-tuned per target |
| Output head | per tokenizer family | fine-tuned |
| Memory gate | init shared | distilled from the target's attention |
| Acceptance thresholds, block-length policy, branch count | — | calibrated |

**Size classes.** Ship 2–3 fixed shapes (for example S ≈ 3M, M ≈ 6M, L ≈ 10M params) so kernels compile once. A new target is a new weights blob, not new kernels.

### 4.2 Target interface contract (what oflm must provide for any target)

```text
target.verify(block_or_tree, mode) ->
    accepted_len, correction_token,
    topk_ids[pos][K], topk_scores[pos][K],      # K ~ 8, quantized
    projected_features[pos][d]                   # adapter output, d = drafter width
target.rollback(to_position)                     # attention: truncate; DeltaNet: replay
target.feature_layers                            # which hidden layers feed the adapter
target.tokenizer_hash, target.model_hash
target.verify_cost(N, context_len)               # measured table, used by the scheduler
```

Rollback is the target's job. For hybrid targets with linear-attention layers, the target runtime keeps per-token (k, v, gate) inputs and replays from the block-start state. It does **not** save a full state copy per draft position. Phase 0 measures this cost for each target.

### 4.3 Drafter package format

```text
drafter-<target>-<size>-v<N>/
  manifest.json      # target model hash, tokenizer hash, size class, precision,
                     # kernel ABI version, feature layers, d, vocab size, slot count,
                     # thresholds, measured acceptance + tok/s on reference hardware
  npu_weights.bin    # packed for the size-class kernel layout
  adapter.<fmt>      # target-device adapter weights (iGPU)
  vocab_map.bin      # drafter id -> target id
  gate.bin           # memory gate params (if enabled)
```

oflm refuses to load a package whose target hash, tokenizer hash or kernel ABI doesn't match.

### 4.4 Onboarding a new target (the per-target fine-tune)

1. Run the phase 0 measurements on the target: verify cost curve, rollback cost, KV share.
2. Capture trajectories (§6.1) from the target on the mixed workload.
3. Build the core vocab set from the target's own outputs. Check coverage (§6.2).
4. Initialize from the family-pretrained core. Train the adapter + head first, then fine-tune everything with the acceptance objective, then QAT at deployment precision.
5. Distill the memory gate from the target's attention.
6. Calibrate thresholds, block-length policy and branch count against the target's measured verify curve.
7. Run the gates in §8. Publish the package with its measured numbers.

Expected cost per target: dominated by trajectory capture (target forward passes), not drafter training (a ~10M-param model).

## 5. Software stack

| Layer | Choice | Notes |
|---|---|---|
| NPU kernels | IRON / MLIR-AIE + AIE C++ kernels, Linux `amdxdna` driver | Needed for tile placement, memory-tile residency and persistent kernels. ONNX Runtime / Ryzen AI EP doesn't give this control. |
| Host ↔ NPU | Persistent kernel polling a doorbell in a shared DDR ring buffer | Fallback: per-cycle dispatch, if the driver can't keep a kernel resident. Phase 1 decides. |
| Reference model / training | PyTorch | Plus a bit-exact emulator of NPU integer numerics. |
| Target runtime | oflm (iGPU) | Implements §4.2. |
| Evaluation | `bench/run_bench.py` (tools, GSM8K, MMLU, ppl) + new drafter timing harness | Same suites as the prune campaigns. |
| Capture / training compute | Rented GPU pods, as in `heal/pod_run.sh` | |

### 5.1 Ring buffer protocol (fixed-size records)

```text
host -> npu  REQ  { seq, accepted_len, correction_tok, n_new,
                    new_token_vecs[n_new][d], projected_features[n_new][d],
                    topk_ids[K], topk_scores[K], slot_updates[], flags }
npu  -> host RESP { seq, n_branches,
                    branch[b] { assumed_k, assumed_correction, tokens[N],
                                q_ids[N][m], q_vals[N][m], confidence[N] } }
```

Both sides use sequence numbers. The host never reads a branch whose `seq` doesn't match its current cycle.

## 6. Data and training

### 6.1 Trajectory capture

For each position on a mixed workload (chat, code, tool calls/JSON, math, long documents), record:
- token ids and accept/reject outcomes from a baseline drafter (DFlash 2 where available)
- the target's top-k (K = 16–32) logits, quantized
- the target's hidden states at the feature layers
- the target's attention mass on old positions from its full-attention layers (gate labels)

**Storage problem.** Raw multi-layer hidden states for a 27B target are ~50 KB per token, or terabytes at 100M tokens. Options, in order of preference:
1. **Online:** run the target live during training on the pod, so nothing is stored.
2. **Fixed compression:** a per-layer PCA/random projection to ~1024 dims in fp8 (~1 KB/token). The adapter learns from that.
3. Fewer feature layers.

**Two data tiers.** Bulk pretraining data can come from the fast full-precision base model (for example Qwen3.8-27B on a GPU). Fine-tune data must come from the exact deployed target (Bonsai 2), because acceptance against the deployed target is ground truth.

### 6.2 Vocab set building

- Rank tokens by frequency **in the target's own outputs** on the workload. Pick the set size from the coverage curve.
- Check the coverage limit: a fully accepted N-block can happen at most cᴺ of the time. Aim for c ≥ 98–99% including context slots.
- Context slots: every distinct token in the current prompt + recent output, up to the slot count, LRU-replaced. The host supplies their embeddings.

### 6.3 Losses

1. **Acceptance objective:** expected accepted length, Σₖ Πᵢ≤ₖ αᵢ with αᵢ = Σₓ min(pᵢ(x), qᵢ(x)), computed on the stored top-k of p. Differentiable.
2. **KD:** forward KL to the target top-k as a stabilizer early in training.
3. **Refinement:** loss on every refine step, weighted toward the last.
4. **Gate:** cross-entropy against keep/merge/dump labels from target attention mass, plus a slot-budget penalty (Lagrangian). Gumbel-softmax during training, hard choices at run time.
5. **QAT** at deployment precision. Ternarize or quantize before the last healing pass.

### 6.4 Training stages

1. Family pretrain of the core on bulk data (tier 1).
2. Per-target: adapter + head warm-up with the core frozen.
3. Per-target: full fine-tune on tier 2 with the acceptance objective.
4. Per-target: train on the drafter's own errors (on-policy refine data), then QAT.

## 7. Phases

Each phase ends with a measurable gate. Change one major variable at a time. Keep every baseline runnable.

### Phase 0 — Timing model and target measurements (no NPU kernels)
- Measure T_verify(N, context) on the iGPU for N = 1…16 and context = 2K / 8K / 32K / 64K.
- Measure rollback cost for the target, including DeltaNet replay if hybrid. Read the target's `config.json` for its attention layout.
- Measure bus contention: target slowdown while a synthetic NPU/CPU load reads DRAM at various rates.
- Measure the baseline drafter (DFlash 2 or best available) acceptance curve against the target.
- Build `sim/timing_model.py`: accepted tok/s ≈ E[accepted + 1] / (T_verify(N) + T_draft_exposed + T_sync), with overlap and branch hit-rate terms.
- **Gate:** report the throughput ceiling for a perfect-speed drafter at baseline acceptance. If the ceiling is under ~1.3× over the best existing setup, stop and rethink.

### Phase 1 — NPU hardware probe
- Usable bytes per compute tile and memory tile after runtime reservations.
- Matmul throughput for M ∈ {1, 4, 8, 16, 32} × common K/N at INT8, INT4, bf16. Unpack cost for ternary → INT8.
- Memory-tile → compute-tile DMA bandwidth, and log-sum-exp merge cost across columns.
- **Persistent kernel:** can a kernel stay resident and poll DDR? Is it preempted or timed out? Doorbell round-trip latency. Per-dispatch latency as the fallback.
- Output: `hardware_profile.json` (the input to the architecture generator).
- **Gate:** a stable host ↔ NPU round trip under ~0.5 ms (persistent or per-dispatch).

### Phase 2 — Reference model, emulator, architecture generator
- PyTorch drafter with all features behind flags: DeltaNet layers, optional attention layer, trimmed head + slots, refine loops with step embedding, memory gate.
- Bit-exact NPU numerics emulator for the chosen precisions.
- Architecture generator: given `hardware_profile.json`, emit configs (width, layers, heads, head size, slot counts, KV budget, precision) that **provably fit**, including buffers. Define the S/M/L size classes.
- **Gate:** emulator matches PyTorch within quantization tolerance. Every emitted config fits in the phase 1 budget.

### Phase 3 — Data pipeline (first target: Bonsai 2)
- Implement §6.1 capture (online or compressed) and §6.2 vocab building.
- Publish the coverage curve and pick the core set size.
- **Gate:** coverage ≥ 98% with slots at the chosen size, and the data loader feeds training at full speed.

### Phase 4 — Drafter v1: one stage, DeltaNet only
- No attention layer, no gate, no refine loop, fixed block length.
- Train (§6.4), then build the NPU kernels for the S size class.
- **Gate (go/no-go for the project):** beats the best conventional drafter in accepted tok/s at equal output, lossless, on the target hardware.

### Phase 5 — oflm runtime integration
- Persistent kernel + ring buffer (§5.1).
- Overlap draft with verify. Branch pre-drafting with a branch budget B.
- Confidence-gated adaptive block length.
- Lossless sampling (§2.2).
- **Gate:** the overlap win shows up in measured tok/s. Report branch hit rate and exposed draft time.

### Phase 6 — Second target onboarding
- Onboard a second target with a **different** size or architecture through §4.4, using the same kernels.
- Freeze the package format and kernel ABI v1.
- **Gate:** the second target ships with only weights and calibration changed, and onboarding effort is recorded. This checks generality before more features are added.

### Phase 7 — Mini-hybrid: long-term attention layer
- Add one full-attention layer with KV split across memory tiles (D5). Sweep the weights/KV split.
- **Gate:** keep it only if acceptance gains beat spending the same bytes on weights, especially on code/tool/long-context slices.

### Phase 8 — Memory gate (keep / merge / dump)
- Baseline: recent window + anchors + top attention-mass tokens.
- Learned gate distilled from target attention, with fixed slot pools per column, log(n) count correction on merged slots, and an optional DRAM cold tier for dumps.
- Measure at 0.5 / 1 / 1.5 MB budgets, plus the "regretted dump" rate.
- **Gate:** beats the heuristic at the same budget.

### Phase 9 — Refinement, trees, relaxed acceptance
- Refine loops R = 1…4 with shared weights + step embedding. Separate Stage-2 weights only if shared weights clearly lose.
- Tree drafts, if oflm supports tree verification.
- Relaxed acceptance modes (top-k match, ratio threshold, typical acceptance), stricter inside code, JSON and digits. Sweep tok/s vs `bench/` scores. Pick the knee. Lossless stays the default.
- **Gate:** each addition is kept only if it raises tok/s (lossless) or improves the tok/s-per-quality curve (lossy).

### Phase 10 — Optional third tier
- Measure the agreement matrix (tiny→middle, middle→target, tiny→target) for middle candidates: a small sibling model, layer-skipped target, or a streamed middle model on the NPU.
- **Gate:** build it only if the timing model predicts > 15% gain.

### Phase 11 — Precision study and hardening
- INT8 vs INT4 vs ternary per size class, including unpack cost and acceptance loss.
- Regression CI: per-package acceptance and tok/s on a fixed prompt set, plus `bench/` scores for lossy modes.

## 8. Metrics and release gates

Report for every package and every phase:

- Accepted tok/s end to end (primary), lossless, and at each lossy setting.
- Mean / median accepted length, full histogram, per-position rejection rate.
- Draft latency per block and per refine loop; **exposed** draft time after overlap; branch hit rate.
- Target verify latency vs N and context; rollback cost.
- Host ↔ NPU round-trip latency; bytes moved per cycle.
- DRAM weight bytes read by the NPU per cycle, and target slowdown from bus contention.
- On-chip occupancy: weights, state, KV, buffers.
- Vocab coverage; regretted-dump rate (if gated); drafter calibration (confidence vs agreement).
- Energy per accepted token.
- Task scores (`bench/`) for any lossy mode; lossless mode must match the target exactly.

**Release gate per target package:** lossless tok/s ≥ 1.2× the best conventional drafter for that target on reference hardware, and no `bench/` regression in lossless mode.

## 9. Proposed repo layout

```
npu-drafter/
  PLAN.md                 # this file
  sim/                    # timing model, arch generator, NPU numerics emulator
  probe/                  # phase 1 hardware probes (IRON/MLIR-AIE)
  model/                  # PyTorch reference drafter
  capture/                # trajectory capture, vocab builder
  train/                  # training stages, QAT, gate distillation
  kernels/                # NPU kernels per size class
  runtime/                # ring buffer protocol, oflm integration glue
  packages/               # manifests + measured results (weights live on HF)
  results/                # timing and acceptance logs, like bench/results
```

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `amdxdna` can't keep a persistent kernel resident | adds dispatch latency every cycle | Per-dispatch fallback. Overlap hides most of it if dispatch is ≪ verify time. |
| Target rollback (hybrid DeltaNet layers) is costly | eats speculative gains on long blocks | Replay from per-token inputs. Limit N via the timing model. |
| Tiny drafter acceptance too low | no speedup | Phase 4 go/no-go. Larger size class. Streamed middle tier (phase 10). |
| Vocab coverage gaps on code/tool workloads | short accepted runs | Context slots; per-target vocab sets. |
| Feature capture storage blows up | slow, costly training | Online capture or fixed compression (§6.1). |
| Per-target fine-tune too expensive | limits targets in oflm | Family pretrain; adapter+head-only fine-tune as a cheap tier. |
| Lossy modes break tool calls / code | silent quality loss | Content-aware strictness, `bench/` gates, lossless default. |
| Planning hardware numbers are wrong | configs don't fit | Phase 1 replaces every figure in §3 before training. |

## 11. Open questions

- Best block length per target: 4, 8, 12, 16 or adaptive? (Phase 0 curve + phase 5.)
- How many refine loops pay off?
- Does the attention layer earn its bytes, or does the target's hidden state carry enough long-range recall?
- Does the learned gate beat the attention-mass heuristic?
- Can the per-target fine-tune be adapter + head only for targets in the same family?
- Ternary vs INT4 once unpack cost is counted?
- How much do verifier top-k / feature feedback improve the next proposal?

## 12. Assumptions to confirm

- The target runs on the iGPU and oflm can append the feature adapter to its forward pass.
- oflm can expose the §4.2 contract, including top-k scores and rollback, for each target.
- Qwen3.8-27B attention layout (hybrid or full) — read from `config.json` in phase 0.
- A GPU runtime exists that runs Bonsai 2 fast enough for trajectory capture. If not, use the full-precision base for tier-1 data and on-device Bonsai for tier-2 data.

## 13. Agent directive

Implement experimentally, not speculatively. Measure the real hardware first (phases 0–1). Keep a runnable baseline at every step. Change one major variable at a time. Treat DFlash 2 as a teacher and benchmark, not a design to copy. Treat acceptance against the deployed target as ground truth. Keep lossless mode as the default and the reference. A feature stays only if it improves accepted tok/s on real hardware, or the tok/s-per-quality curve for lossy modes.
