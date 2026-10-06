# Layer merging: a second shot at 30 → 20

The second layer cut (C2) on the healed 30-layer model was fatal: tools .05, ppl 3,741.
It dropped GDN slot 1 of every `[GDN, GDN, attn]` group. This experiment makes the same
cut, to the same 20-layer `[GDN, attn] × 10` layout, but folds the dropped layer into
its neighbour instead of throwing it away. Then it heals as before.

Script: `merge_layers_qwen35.py`. Test: `merge_smoke.py` (about 15 s on CPU).

## Why not just average the two layers

There are two reasons.

1. **Internal units don't line up.** GDN head 3 in layer a has nothing to do with GDN
   head 3 in layer b. Expert 17 in layer a has nothing to do with expert 17 in layer b.
   Averaging by index blends unrelated parts.
2. **An average does half the work.** Two layers add both of their updates to the
   residual stream. An average adds roughly half of each.

Naive averaging (`avg:avg:none`) is still in the menu, so the numbers can settle it.

One correction to what I said earlier: LaCo's merge rule, `θ_a + Σ(θ_k − θ_a)`, turns
into "keep layer b" when you merge only two layers. It doesn't help for pairs, so it
isn't used here.

## What gets built instead

Each layer is a token mixer (GDN) followed by an MoE. Each part gets its own treatment.

| part | treatment | why it can work |
|---|---|---|
| **MoE: union** | Pool both layers' experts (512) and keep the 256 that carry the most routed output on calibration data. Their router rows come along. | An expert is a whole function from the residual stream back into it, so no neuron matching is needed. Each layer's norm scale is folded into its own experts, so they can share one input. |
| **Shared experts: concat** | 512 + 512 → one 1024-wide shared expert. | This is an exact sum, except that the two now share one gate. |
| **Output map: fit** | A ridge least-squares map from [routed out, shared out] to what the original pair added to the residual stream. It is folded into every `down_proj`. | This puts back the update size lost when 16 expert slots become 8. It is the ReplaceMe idea (a training-free linear fix), applied where it folds exactly. |
| **Mixer** | Keep a, keep b, or average. | Two GDNs can't be stacked into one, so the fit has to cover what is lost. |

The calibration is **sequential**. Each group's inputs come from the already-merged
groups before it, so every fit also corrects drift from earlier merges. The targets
always come from the original model.

For each pair, every variant gets a score on held-out sequences. The best one is kept,
and a plain drop is one of the choices. So no group should come out worse than C2
(measured locally; see the caveats). The run ends with a held-out perplexity for three
streams: the teacher, the merged model, and plain C2. You get the "does merging beat
dropping" answer before converting anything.

## Run it

```
python merge_layers_qwen35.py merged-heal-c merged-c2 \
    --calib heal-artifacts/calibration.txt
```

* Add `--dry-run` to score and report without writing a checkpoint.
* The full fit has 4096 inputs per output channel, so it can overfit. Each fit tries
  several ridge strengths (`--ridge 0.001,0.01,0.1,1`) and keeps the best one on
  held-out data. More calibration (`--n-seq`) still helps the most.
* `merge_report.json` in the output holds every variant's held-out error for every group.

Then follow the same path as `c2_chain.sh`: convert → Q4_K_M → bench. Heal only if
the unhealed numbers beat C2 by a clear margin.

```
python heal/train_heal.py --student merged-c2 --data ... --out runs/heal-c2m --loss sft --four-bit
```

## What to watch

* **Pre-heal ppl, merged vs. drop.** This is the first go/no-go signal. If merged
  isn't clearly below drop, the fits aren't buying anything on this architecture.
* **Which variants win.** If `a:a:none` (plain drop) wins most groups, merging doesn't
  help here. If `*:union:full` wins, the expert pool and output fit are doing the work.
* **Held-out vs. train error.** A big gap means the full fit is overfitting. Use more
  calibration, a higher `--ridge`, or `diag` variants.
* **Speed.** The concatenated shared expert adds about 3M active params per layer. That
  is about 1% of active params, and the other layers carry zero padding to match. With
  `--top-k 10` you get back some lost expert compute at a speed cost. Benchmark both.

## Caveats

* Local error isn't end-to-end quality. The held-out ppl is the better signal, and the
  bench is the real one.
* llama.cpp hasn't been tried yet with a 1024-wide shared expert. 1024 fits Q4_K's
  256-element blocks, and every layer and the MTP head share one width. Still, check that
  the convert step succeeds. `--shared first` keeps the width at 512 if it fails.
* Calibration text is general prose. Tool-call traffic in the calibration set would
  steer expert selection toward what the bench scores.
* The MTP head was trained on the 30-layer model's final hidden states. Expect a lower
  draft acceptance rate until healing.
