# Shrinking Qwen3.6-35B-A3B into Qwen3.6-27B-A2.8B

*Project log, August 2026. Target hardware: a laptop with an 890M iGPU and 96 GB system RAM. Training ran on rented H200s. Total cloud spend: about $80.*

**Model weights:** [huggingface.co/Cyronius/Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B)

**Reproduce it:**
```
git clone https://github.com/Cyronius/qwen-prune-heal-pipeline.git
cd qwen-prune-heal-pipeline

# 1. prune a 40-layer Qwen3.6-35B-A3B checkpoint
python prune_qwen35.py --skip-expert-cut   # layer cut only (the shipped path)

# 2. smoke-test the healing pipeline on CPU before renting a GPU
cd heal && python smoke.py --work <scratch dir>

# 3. heal on a GPU box — see heal/README.md for the full LoRA run
pip install -r heal/requirements.txt
```
Converting to GGUF needs `patches/llamacpp-qwen35-tokenizer-hash.patch` applied to llama.cpp first — see `patches/README.md`.

**Repo structure:**
| path | what's there |
|---|---|
| `prune_qwen35.py` | the layer-cut / expert-width-cut script |
| `heal/` | LoRA healing pipeline: teacher generation, training, merging, CPU smoke test |
| `patches/` | the llama.cpp tokenizer-hash patch needed to convert this model family |
| `bench/` | benchmark harness, tool-calling cases, and the raw result logs behind every number below |

Model weights, adapter checkpoints, and quantized builds aren't in this repo — see the Hugging Face link above.

---

**Why an MoE instead of the newer dense Qwen3.8-27B?**

Because I care about decode speed on shared-memory hardware, and on this machine decode is mostly a memory-bandwidth problem.

With a dense 27B model, roughly all 27B parameters have to be touched for every token. Qwen3.6-35B-A3B is different: it has about 35B parameters in total, but only around 3.45B are active for any given token.

That makes MoE much more interesting for this experiment. I can reduce the amount of model that gets used per token without throwing away the same proportion of the model's total capacity. With a dense model, those two things are basically tied together. That's why I started with the MoE checkpoint.


## TL;DR

I removed 10 of 40 layers from Qwen3.6-35B-A3B, cutting the language model from about **34.7B total / 3.45B active parameters** to **26.2B / 2.83B active**. The damaged model was still usable, and a $53 LoRA healing run brought most of its quality back. On my tool-calling benchmark it actually finished ahead of my base Q4 model. And the cuts gave a consistent 15-30% decode speedup from the layer cut alone.

I'm shipping two builds of **Qwen3.6-27B-A2.8B**: a balanced 16 GB Q4_K_M and a 12 GB speed-demon variant with 3-bit routed experts. Both consistently beat a same-lineage base model with MTP added that shares everything except the pruning and healing steps. Exact tok/s numbers moved around enough between repeat tests that I'm reporting the comparison, not a specific figure.

---

## What I was working with

A few details mattered before cutting anything.

Qwen3.6-35B-A3B has 40 language-model layers arranged as 30 gated-DeltaNet layers and 10 full-attention layers. Each layer also owns a full set of 256 routed experts plus a shared expert, so removing a layer removes much more than just its attention machinery. The checkpoint also contains a vision tower and a one-layer MTP head.

I also recalculated the parameter counts directly from the checkpoint. The language-model core is about **34.66B total parameters and 3.45B active per token**. After pruning, that becomes **26.23B total and 2.83B active**. Including the vision tower brings the pruned model to about 26.7B total, which is why I call it **Qwen3.6-27B-A2.8B** rather than pretending it is still a 35B model.

## Choosing what to cut

I looked at several ways to shrink it.

| approach | result |
|---|---|
| Halve expert width from 512 to 256 | Survived, but badly damaged tool selection and math. Stock LoRA also can't directly repair the routed-expert tensors. |
| Halve attention head dimension | Not worth it. Attention is too small a fraction of active parameters, and the geometry gets awkward fast. |
| Remove 1 of every 3 GDN layers | Best option. Reduced 40 layers to 30 while preserving a legal attention pattern. This became the shipped model. |
| 4:2 structured sparsity | Scrapped. A custom Vulkan implementation looked unlikely to beat llama.cpp's existing dense Q4 kernels by enough to justify the work. |
| Trim vocabulary | Deferred. The theoretical gain was small and aggressive trimming can backfire by increasing token counts. |

Before training anything, I did the deliberately crude experiment: prune each candidate and benchmark it immediately.

| model | GB | tools | GSM8K | MMLU | loop | ppl |
|---|---:|---:|---:|---:|---:|---:|
| base Q4_K_M | 21.2 | .90 | .93 | .77 | .02 | 5.50 |
| layer cut only | 16.0 | .70 | .80 | .67 | .09 | 10.84 |
| expert-width cut only | 11.4 | .15 | .07 | .77 | .22 | 20.33 |
| both cuts | 8.6 | .08 | .00 | .17 | .58 | 101.3 |


- **tools**: accuracy on a set of agentic tool-calling cases (`bench/cases/tools.jsonl`),
  does the model call the right tool with the right arguments, or correctly decline
  to call one at all?
- **gsm8k**: accuracy on GSM8K, the standard grade-school math word-problem benchmark,
  scored on whether the final numeric answer matches.
- **mmlu**: accuracy on MMLU, a multiple-choice knowledge benchmark spanning many
  academic subjects, scored on whether the model picks the right letter.
- **loop**: the loop rate, how often the model runs out its token budget without ever
  producing a tool call or any real content, i.e. it gets stuck generating nothing
  until the server cuts it off. Lower is better; this is a distinct failure mode from
  just answering wrong.
- **wikitext ppl**: perplexity, a measure of how well the model predicts ordinary text.
  At each word, the model is effectively guessing among some number of plausible next
  words; perplexity is roughly that number. A perplexity of 5 means the model is about
  as unsure as if it were choosing among 5 options at each step; a perplexity of 100
  means it's badly lost. Lower is better. Measured on 32 chunks of the wikitext-2 test
  set with llama.cpp's perplexity tool at context length 512. This tracks general
  language ability, not any one task, so it catches damage that task accuracy misses.


The layer cut hurt almost everything, but not catastrophically. The expert-width cut was stranger: MMLU stayed at base level while tool use and math collapsed. The model still knew things; it had lost some ability to act on them.

The biggest problem was that cuts interacted badly. Two individually survivable cuts produced a nearly useless model when stacked. Perplexity also tracked the damage extremely well, which made it useful as a cheap healing signal.

## Healing the 30-layer model

I trained on about **18,000 prompts / 16 million supervised tokens**: roughly 55% general text, 35% tool-use examples, and 10% GSM8K-style reasoning.

The tool data was deliberately separated from evaluation. None of the seven evaluation tool schemas appeared in training. I mixed hand-written tools with thousands of ToolACE schemas so the model had to learn tool-use behavior rather than memorize a tiny schema set.

The healing itself used rank-32 LoRA, about 33 million trainable parameters, or only 0.125% of the model. Plain supervised fine-tuning against teacher outputs turned out to be enough; I built a more elaborate top-k distillation path, but never needed it.

The most useful engineering work was a CPU smoke test that exercised the real tokenizer and the whole save/load pipeline before renting a GPU. It caught six bugs, including two that could have silently ruined the final model: one loader path dropped the vision tower, and another dropped all 19 MTP tensors. It also caught a one-token target-alignment bug that still produced a perfectly normal-looking training loss curve.

The actual H200 training run cost about **$53**. It ran 550 steps at roughly 770 tokens/sec. Held-out perplexity improved steadily at every checkpoint.

After training, I merged the LoRA back into the model, manually restored the MTP tensors, converted to GGUF, built an imatrix for quantization, and produced the Q4_K_M release.

### Healed result

| metric | healed 27B | base | unhealed 27B |
|---|---:|---:|---:|
| tools | **.975** | .90 | .70 |
| GSM8K | .867 | .933 | .80 |
| MMLU | **.767** | .767 | .667 |
| loop rate | **.024** | .024 | .094 |
| perplexity | 8.01 | 5.50 | 10.84 |
| size | 15.95 GB | 21.17 GB | 15.95 GB |

Tool calling recovered beyond the base model, and MMLU and loop rate returned to base level. GSM8K remained a little behind, and perplexity recovered most of the way but not completely.

The run only covered about half an epoch, so I probably could have closed more of that gap by spending another ~$30. I stopped because the model was already useful and the bigger unresolved question was speed.

## Where the real speedup came from

I also tried reducing the number of routed experts used per token below the model's default `k=8`, hoping fewer active experts would mean less data moved per token.

| k | tools | GSM8K | MMLU |
|---|---:|---:|---:|
| 8 | .975 | .867 | .767 |
| 6 | .925 | .867 | .767 |
| 4 | .900 | .733 | .700 |

`k=6` held accuracy almost exactly, missing two more tool cases; `k=4` started losing GSM8K and MMLU too. I originally reported a large decode speedup from dropping to `k=6` (about 46% faster). It didn't hold up: re-testing the identical file and flags three times on the same day gave three different, contradicting answers — k=6 slower than k=8, then indistinguishable, then faster. I can't get a stable number here, so I'm not giving one. Treat the speed effect of lowering `k` as unresolved on this hardware. More benchmarking will be needed to get a result we can trust.

MTP — llama.cpp's implementation of multi-token prediction, a small auxiliary head that drafts several tokens ahead for the main model to verify — is a separate, better-established piece of technology, and I'm not trying to re-litigate it here beyond what this project showed. It's worth shipping on: tool-call decode came out flat-to-positive with MTP enabled in every measurement I took, consistent with MTP working best on predictable output. Ordinary chat decode was noisier — usually faster with MTP on, occasionally not — so I'm not quoting an exact percentage for either.

Combining reduced `k` with MTP looked bad in my first pass (my hypothesis at the time: the draft head is trained against the model's normal k=8 behavior, so changing the routing makes its guesses match less often). Given that the `k`-alone numbers didn't reproduce either, I no longer trust that specific story enough to repeat it as a finding — just note that I never found a reduced-`k` config worth shipping over plain `k=8 + MTP`.

There is one small caveat to the usual claim that speculative decoding is perfectly lossless. In a 40-case confirmation run, MTP scored 37/40 where plain decoding scored 39/40. Batched verification changes floating-point math slightly, so very close token decisions can occasionally flip. Functionally it is very close, but not bit-for-bit identical on this stack.

## Dead ends worth recording

A few failures changed how I think about model surgery.

### Narrowing experts again

I tried a gentler expert-width cut, 512 down to 384, on top of the healed model. Quality collapsed again: GSM8K fell to .27, MMLU to .43, and perplexity rose to 13.2.

It also ran into a completely different problem: the k-quant formats I was using require matrix dimensions compatible with 256-element blocks. Width 384 forced important expert tensors into a much larger fallback format and decode dropped to **9.4 tok/s**. Quantization format constraints turned out to be architecture constraints in practice.

### Merging supposedly redundant channels

Instead of deleting weak expert channels, I tested whether highly similar channels could be merged.

Using real expert activations across 24,000 tokens, I looked at 764 experts. Only **0.2% of channel pairs** had correlation above 0.9, and the typical channel's best match was only about 0.35.

In other words, there wasn't much redundancy to exploit. The width cut wasn't throwing away duplicate capacity; it was throwing away real capacity.

### Cutting layers a second time

I also tried going from 30 layers down to 20. That model was dead: tool accuracy fell to .05 and perplexity exploded to 3,741. The first layer cut was about as far as this approach could go.

## A speed-demon build, for when speed matters most

I pushed the quantization further to build a second, faster variant: roughly 3-bit routed experts plus a 4-bit `lm_head`, leaving MTP, attention, and GDN tensors alone.

| model | size | tools | GSM8K | MMLU | ppl | decode | tool decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced 27B | 16.49 GB | .95 | .80 | .733 | 8.01 | 36.8 | 40.7 |
| speed-demon 27B | **12.01 GB** | .90 | .733 | .767 | 8.29 | **40.7** | **44.1** |
| base Q4 | 21.17 GB | .90 | .933 | .767 | 5.50 | 27.8 | 27.9 |


* **decode** — Ordinary generation speed in tokens per second.
* **tool decode** — Generation speed while producing tool-call output, which tends to be more predictable and benefits more from MTP/speculative decoding.


The file gets 27% smaller and decode gets about 11% faster, at the cost of some tool accuracy and GSM8K, plus a worse loop rate. I'm shipping both: the balanced build for the best overall quality, and the speed-demon build for setups where memory or throughput matters more than the last few points of accuracy. TBH, the speed demon is my choice.

## What I took away from it

1. **Benchmark the broken model before trying to heal it.** Cheap destructive experiments tell you which change caused which failure.
2. **Architectural cuts interact.** Two acceptable cuts can combine into a disaster.
3. **Perplexity is a useful damage gauge.** It caught general degradation that task benchmarks alone would have missed.
4. **Smoke-test the entire model pipeline locally.** Silent loader behavior is dangerous; the MTP head and vision tower would both have disappeared without explicit checks.
5. **Quantization rules constrain architecture.** A mathematically reasonable width can still be a terrible practical choice if the inference format hates it.
6. **A single same-session speed test still isn't enough.** My first layer-cut speed measurement looked clean and controlled and was still wrong by a wide margin. It took re-running the same comparison, and checking it against unrelated pairs, to catch it.

The result is a model about 25% smaller than the one I started with, with most of the lost quality recovered. The layer cut did produce a real decode improvement, just not the dramatic one I first thought I had measured. MTP did the rest.

I'm shipping two builds: the balanced 16 GB **Qwen3.6-27B-A2.8B Q4_K_M with k=8 and MTP enabled**, and the 12 GB speed-demon build for when speed and memory matter more than the last few points of accuracy. If I push this further, retraining the MTP head is probably the next experiment worth paying for.

## What's next

There are still a few experiments I'd like to run: finish the remaining healing epoch, retrain the MTP head, try a custom adapter for the routed experts, and run larger benchmark suites.

The next healing run is small — probably around **$30 of H200 time** — but some of the follow-up work could use substantially more compute.

If you have spare H100/H200/B200 time or cloud credits, or want to help cover a run: [PayPal](https://paypal.me/JAttoun). Any results, including failures, will stay public along with the code and checkpoints I can redistribute.

## License

Code in this repo is Apache-2.0 — see [LICENSE](LICENSE). Model weights are distributed separately on Hugging Face under their own license terms.
