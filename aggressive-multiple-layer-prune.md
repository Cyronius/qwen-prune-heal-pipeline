Goal: shrink Qwen3.5-35B-A3B toward faster tok/s for agentic/tool-calling loops on pleurotus (890M iGPU), without wrecking tool-call reliability.

Revised 2026-08-04 after evaluation (.claude/plans/shrink-plan-evaluation.md): cut B dropped,
4:2 sparsity plan (weight-pruning.md) scrapped, projections corrected against published specs.

BASELINE (verified against published architecture specs)
- Current: ~35B total / ~3B active, Q4, 20-30 tok/s on pleurotus
- 40 layers = 30 GDN + 10 gated attention (3:1 interleave)
- Attention: GQA 16 Q-heads : 2 KV-heads, head_dim 256, 64-dim partial RoPE
- MoE: 256 experts, top-8 + 1 shared, intermediate 512, hidden 2048
- Active-param split per token: experts ~1.13B, GDN ~0.8B, attention ~0.27B, embed+lm_head ~0.6B
- Still verify actual safetensors shapes before surgery, but the split above supersedes the old
  back-of-envelope (which overstated attention 3-4x at ~1.0B)

THE FIXED FLOOR (why every projection below is capped)
The lm_head is a dense 2048 x 151k matmul (~0.31B params) read in full every token -- no routing,
no sparsity. No cut here touches it, so it grows from ~10% of bytes/token now to ~14% after the
cuts land. All speedups are computed against shrinkable bytes only. (Vocab trimming could shave
this but is a ~1.05x lever at best -- see REJECTED / DEFERRED.)

TWO STRUCTURAL CUTS (cut B is dead -- see REJECTED / DEFERRED)
A. Expert width: halve intermediate dim 512 -> 256 (shrinks every expert, not expert count --
   distinct from expert merging). Saves ~0.57B active. Experts are ~32B of the 35B total, so this
   is where nearly all the storage win lives. Also the riskiest cut: it removes ~50% of the
   model's knowledge-storage capacity. Published recoveries of cuts this size (Minitron) used
   distillation over tens of billions of tokens; a $20-100 QLoRA pass is 1-2 orders of magnitude
   less signal. Expect tool-call formatting/selection/typing to recover well (the dataset targets
   exactly that traffic); expect broad knowledge and multi-step reasoning to take a real hit.
C. GDN depth: prune ~25% of the 30 GDN layers (leave the 10 attention layers alone). Saves ~0.2B
   active. Cheapest, safest cut -- do it first; it validates the whole heal-and-eval pipeline.

PROJECTED IMPACT (corrected)
- Cut C alone: active ~3B -> ~2.8B, ~1.15-1.25x -> ~23-35 tok/s. Stock llama.cpp.
- Cuts A+C combined: active ~3B -> ~2.2B, ~1.3-1.5x -> ~27-40 tok/s. Stock llama.cpp.
  (The old ~1.7x / 35-50 tok/s projection rested on the inflated attention estimate + cut B.)
- Q4 storage: ~17.5GB -> ~10GB with cut A. The freed RAM is headroom for context/KV cache, which
  may matter as much as tok/s for agentic loops.

TEST SEQUENCE (staged, not all-at-once)
1. Generate one distillation dataset from the original unpruned model -- teacher outputs (ideally
   logits, not just text) over real agentic/tool-calling traffic: actual tool schemas, multi-turn
   loops, and negative cases (when NOT to call a tool). Runs locally overnight via llama-server.
   Reuse this same dataset for every healing run below. Build the eval set at the same time:
   valid JSON rate, correct function selection, correct argument types, AND reasoning-dependent
   tool use (when not to call, multi-step planning) -- not just format checks, and not perplexity.
2. Apply cut C alone -> heal (QLoRA-style pass, rented GPU, ~$20-100) -> benchmark. This is a
   shippable milestone on its own and the cheapest way to prove the pipeline end-to-end.
3. Apply cut A alone (to a fresh copy) -> heal -> benchmark. Diagnostic checkpoint: this run
   tells you whether expert-width damage is recoverable at this healing budget before committing
   to the combined model. Budget more than $100 here if the eval shows damage -- the healing
   budget is this plan's most optimistic number.
4. If both look survivable: apply A+C together to a FRESH copy of the original (unhealed) model,
   then run ONE new healing pass on the combined model. Don't merge the separately-healed
   models -- each one's adapters corrected for a different specific break; merging risks
   interference. The combined run ships; the individual runs are diagnostic blame-assignment.
5. Quantize LAST, after healing, using the Unsloth dynamic 4-bit approach. Pruning and healing
   need higher-precision gradients; quantizing first stacks precision loss under the healing pass.

REJECTED / DEFERRED (decided 2026-08-04 -- don't relitigate without new data)
- Cut B (halve attention head_dim): DEAD. Attention is only ~0.27B active (~9%, not the ~33%
  originally assumed). KV is already at the GQA floor (2 KV-heads across only 10 attention
  layers), so there is no free KV win. Slicing head_dim 256 -> 128 destroys pretrained Q/K
  geometry including the partial-RoPE structure -- hardest damage to heal, smallest prize.
- Plan B (4:2 structured sparsity + custom Vulkan kernel, weight-pruning.md): SCRAPPED as
  infeasible. Expert matmuls go through mul_mat_id (runtime per-token expert dispatch); the
  bucketed sparse format multiplies that irregular dispatch by 6 sub-matrices and has to beat
  llama.cpp's heavily-optimized dense Q4 path to show any win. Ceiling ~1.15-1.25x for
  weeks-to-months of kernel work plus a permanent llama.cpp fork. Lower ceiling than cut A at
  ~10x the effort. Do not combine its ideas with cut A regardless -- both remove ~50% of the
  same expert weights; stacked = ~75% expert reduction, past the flat-then-cliff point.
- Vocab trimming (shrink embed + lm_head): DEFERRED. The only lever that touches the lm_head
  floor, but a small one: trimming 151k -> ~50k is ~1.05x net after token-count inflation, and
  aggressive trims (e.g. 10k) go NEGATIVE -- per-token savings hit diminishing returns (~2k
  params/row) while common-token fragmentation inflates every sequence ~25-40%, shrinking the
  effective context window and slowing real-text throughput. If ever revisited: both-sides trim
  keeps stock llama.cpp but needs BPE merge-closure surgery + healing on re-tokenized data;
  output-side-only is safer but needs a small loader patch. Not worth it initially.
- Speculative decoding: NOT rejected -- it's the one lever that amortizes the ENTIRE per-token
  read including the lm_head floor, works in stock llama.cpp today, and requires no model
  surgery. Orthogonal to the cuts; worth benchmarking with a small draft model independently of
  this plan.

OPEN ITEMS BEFORE STARTING
- Pull real tensor shapes from the safetensors file to confirm the verified split above
- Check the actual GDN state-dimension implementation before assuming layer removal works cleanly
  (SSM-style layers can be sensitive in ways attention isn't)
- Decide GDN layer selection method (redundancy scoring within GDN layers specifically, not
  pooled with attention layers)
- Line up a rented GPU (40-80GB class) for the QLoRA passes -- pleurotus can't run training
