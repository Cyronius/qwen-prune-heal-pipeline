# heal/

Scripts for repairing a pruned Qwen3.6-35B-A3B checkpoint with LoRA.

The plan these implement is `.claude/plans/heal-pruned-qwen36.md`. Read that first for
why any of this is shaped the way it is.

## Files

| file | what it does |
|---|---|
| `common.py` | LoRA target regex, model loading, the tiny test config |
| `tools_train.py` | the hand-written training catalog: 72 tools, 18 domains |
| `toolace.py` | schemas and prompts from the ToolACE dataset, for schema breadth |
| `make_prompts.py` | assembles the prompt pool from both sources |
| `gen_teacher.py` | records teacher behaviour into top-k log-prob shards |
| `train_heal.py` | LoRA training, SFT or KD loss |
| `merge_heal.py` | folds adapters back in and reattaches the MTP head |
| `smoke.py` | runs all of the above on CPU at toy scale |

## The training tool catalog

`tools_train.py` is deliberately disjoint from `bench/tools_catalog.json`. No tool here
shares a name or a schema with the seven the benchmark scores on, and `make_prompts.py`
refuses to run if that ever stops being true.

That choice costs a little accuracy and buys the ability to interpret the result. If the
model trained on the same seven schemas it is scored on, a higher `tools_acc` would not
tell you whether tool-calling generalised or whether those seven got memorised.

Run it directly to inspect it:

```
python tools_train.py
```

Structure: 18 domains, four tools each, with at least one confusable pair per domain.
Confusable pairs matter because Round-1's damage was to tool *selection*, not tool
formatting. A catalog of unambiguous tools would not exercise the broken behaviour.

Prompts are generated in the same five categories the benchmark scores, weighted
towards where the damage was:

| category | share | what it tests |
|---|---|---|
| selection | 30% | several confusable tools visible, one correct |
| simple | 24% | one obviously-right tool |
| argtype | 20% | correct types, enums, required fields |
| negative | 16% | no tool applies; calling anything is wrong |
| multiturn | 10% | arguments carried over from an earlier turn |

Request text comes from templates with `{slot}` placeholders filled from large value
pools, so distinct prompts are combinatorial rather than capped at the template count.

## Why ToolACE is mixed in

72 hand-written tools is enough for prompt variety and nowhere near enough for schema
variety. At 7,000 agentic prompts each of those 72 schemas gets shown about 280 times.

The benchmark scores transfer to seven schemas the model has never seen. Training on a
small closed set risks teaching *these 72 tools* rather than *how to read a function
definition* — which excluding the eval tools does not prevent on its own.

`toolace.py` adds 15,870 distinct schemas from
[Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Apache 2.0,
ungated, one 37 MB file). `--toolace-frac` controls the split, defaulting to 0.6.

| pool | distinct prompts | distinct schemas | impressions per schema |
|---|---|---|---|
| hand-written only | 87.6% | 72 | 280 |
| mixed, default 0.6 | 97.2% | 9,266 | 2.3 |

Both sources are kept because they do different jobs. ToolACE's schemas are
RapidAPI-flavoured — sports, finance, crypto — which is a different distribution from
the benchmark's everyday tools. The hand-written catalog keeps the training mix close to
what gets scored; ToolACE stops the tool set being memorisable.

**We take only ToolACE's schemas and user turns.** Every assistant turn is discarded.
Their reply format is a bespoke DSL, `[Func(arg="value")]`, not the JSON tool calls our
chat template emits — and the point of this pipeline is to distil our own teacher, not
somebody else's model. Their replies are used solely to label a row positive or negative.

Three format quirks are handled in `toolace.py`: half the tool names contain spaces and
are slugged, parameter objects are typed `dict` rather than `object`, and each tool
carries a stray top-level `required: null`.

ToolACE does contain `create_reminder`, `send_email` and a weather tool that slugs to
`get_weather`. Those are filtered out. `make_prompts.py` also asserts across the whole
written pool, so the guard does not depend on one producer behaving.

## Run the smoke test first

It builds a randomly-initialised model with the same architecture, the same layer
pattern and the real tokenizer, then drives the whole pipeline through it.

```
cd heal
python smoke.py --work <some scratch dir>
```

It takes about two minutes on CPU and needs no GPU. Run it after any change to these
scripts. Every minute it costs here is a minute not spent debugging on a rented GPU.

## The real run

On the rented box:

```
pip install -r requirements.txt

# 1. general slice: teacher-force over existing text
python gen_teacher.py --teacher <base model> --mode forward \
    --input prompts-general.jsonl --out data/general

# 2. agentic and reasoning slices: sample from the teacher
python gen_teacher.py --teacher <base model> --mode generate \
    --input prompts-agentic.jsonl --out data/agentic

# 3. train
python train_heal.py --student ../pruned36-c-only \
    --data data/general data/agentic --out runs/heal-c --loss sft --four-bit

# 4. merge
python merge_heal.py --student ../pruned36-c-only \
    --adapter runs/heal-c/adapter-final --out merged/heal-c
```

Then convert to GGUF (apply `patches/llamacpp-qwen35-tokenizer-hash.patch` first),
quantise with imatrix, add the result to `bench/registry.json`, and run
`bench/run_bench.py`.

## Five traps, all found by running the code rather than reasoning about it

**KD targets can be silently misaligned.**
For a generated sequence, the teacher records a distribution only for the steps it
sampled. Those start at the prompt boundary, not at position 0. Writing them at
position 0 pairs every teacher distribution with the wrong student position.

The loss curve still falls smoothly when this is wrong. The model just learns the wrong
thing. `smoke.py` checks the alignment directly by constructing rows whose top-1 teacher
token is the true next token, then asserting that holds after collation.

**A prompt longer than `--max-len` yields a NaN loss.**
Truncation can leave a sequence with zero supervised positions. Cross-entropy over zero
positions is NaN, and one such microbatch poisons the whole gradient accumulation group.
`train_heal.py` drops those sequences up front and reports how many. Set `--max-len`
above your longest prompt; serialised tool schemas are long.

## Three more traps

**Loading through the wrong class silently deletes things.**
`AutoModelForCausalLM` resolves to `Qwen3_5MoeForCausalLM`, which has no vision tower.
It loads a checkpoint containing `model.visual.*` without complaint and saves one
without them. Nothing warns you.

These scripts pick the class explicitly via `common.student_class()`. The default is
`Qwen3_5MoeForConditionalGeneration`, so the healed checkpoint keeps the same structure
as the base we benchmarked against. Pass `--text-only` to drop the vision tower on
purpose. Whatever you choose, use the same setting for training and for merging.

**The MTP head is dropped no matter which class you use.**
`Qwen3_5MoePreTrainedModel` sets `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`.
`merge_heal.py` copies those 19 tensors across by hand. `smoke.py` asserts they
survived, because losing them costs decode speed and produces no error message.

**The module tree differs between the two classes.**

| class | prefix |
|---|---|
| `Qwen3_5MoeForConditionalGeneration` | `model.language_model.layers.N.` |
| `Qwen3_5MoeForCausalLM` | `model.layers.N.` |

`LORA_TARGET_REGEX` makes the `language_model.` segment optional so it matches both.
The vision tower's own Linear layers sit under `model.visual.blocks.N.`, so
`layers.\d+` never reaches them.

## What is not trainable here

The routed experts (`gate_up_proj`, `down_proj`) and the router (`gate.weight`) are 3-D
and 2-D `nn.Parameter` tensors, not `nn.Linear` modules. PEFT cannot attach LoRA to
them.

That is fine for cut C, which leaves the experts untouched. It is a blocker for cut A,
whose damage is inside those exact tensors. See section 8 of the plan.

## Speed

Always pass `--experts-impl grouped_mm` on GPU. The default expert forward is a Python
loop over 256 experts calling `F.linear` once each. It is a correctness reference, not
a training kernel.

Pass `--experts-impl ""` on CPU, where `grouped_mm` has no fast path anyway. `smoke.py`
does this for you.
