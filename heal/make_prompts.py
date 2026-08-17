"""Assemble the prompt pool that gen_teacher.py runs over.

Three slices, mixed in the proportions the plan calls for.

  general    Existing text, used with gen_teacher.py --mode forward. Restores
             perplexity and knowledge, which cut C damaged as much as it damaged
             tool behaviour.
  agentic    Tool-calling requests, used with --mode generate. Includes negative
             cases, where the right answer is to not call a tool.
  reasoning  Multi-step word problems, used with --mode generate.

Contamination guard: the benchmark's own cases are excluded by construction. The tool
cases in bench/cases/tools.jsonl are never read here, and the GSM8K questions in
bench/data/gsm8k.jsonl are subtracted from the reasoning slice. Training on either
would make the Round-2 numbers meaningless.

Usage:
  python make_prompts.py --out prompts --total 20000
  python make_prompts.py --out prompts --total 20000 --general-dataset HuggingFaceFW/fineweb-edu
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import read_jsonl, write_jsonl
from tools_train import (BY_DOMAIN, EVAL_TOOL_NAMES, TOOLS, build_multiturn,
                         render_negative, tools_with_follow_ups)

HERE = Path(__file__).parent
BENCH = HERE.parent / "bench"

MIX = {"general": 0.55, "agentic": 0.35, "reasoning": 0.10}

# The five shapes the benchmark scores, so the training data exercises the same
# behaviours. Proportions are weighted towards selection and negatives, which is where
# Round-1 showed the damage: cut A refused to call tools rather than mis-formatting them.
CATEGORY_MIX = {
    "simple": 0.24,      # one obviously-right tool visible
    "argtype": 0.20,     # correct types, enums and required fields
    "selection": 0.30,   # several confusable tools visible, one correct
    "negative": 0.16,    # no tool applies; calling anything is wrong
    "multiturn": 0.10,   # the arguments depend on an earlier turn
}

# Openers that vary the surface form without changing what is being asked.
FRAMINGS = [
    "{ask}",
    "Quick one: {ask}",
    "When you get a moment, {ask_lower}",
    "{ask} Keep it brief.",
    "Hey, {ask_lower}",
    "I need this done: {ask}",
    "{ask} Thanks.",
]

# Nudges that force explicit values into arguments, for the argtype slice.
ARGTYPE_NUDGES = [
    "Use the exact values I gave.",
    "Don't guess any of the optional fields.",
    "Fill in every field you can from what I said.",
    "Be precise about the numbers.",
]


def gsm8k_eval_questions():
    """The questions already used for evaluation. Never train on these."""
    path = BENCH / "data" / "gsm8k.jsonl"
    if not path.exists():
        return set()
    return {r["question"].strip() for r in read_jsonl(path)}


def build_general(n, dataset, split, text_field, seed):
    """Pull raw text. Falls back to a local jsonl if datasets is unavailable."""
    if Path(dataset).exists():
        rows = read_jsonl(dataset)
        texts = [r.get(text_field) or r.get("text") for r in rows]
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split, streaming=True)
        texts = []
        for row in ds:
            texts.append(row[text_field])
            if len(texts) >= n:
                break
    rng = random.Random(seed)
    rng.shuffle(texts)
    return [{"slice": "general", "text": t} for t in texts[:n] if t and t.strip()]


def frame(ask, rng):
    shape = rng.choice(FRAMINGS)
    return shape.format(ask=ask, ask_lower=ask[0].lower() + ask[1:])


def visible_tools(tool, rng, min_extra, max_extra, prefer_siblings):
    """Choose which tool schemas the model gets to see alongside the right one.

    Siblings are the other tools in the same domain. Showing them is what makes a case
    a selection test rather than a formatting test: schedule_meeting next to
    add_personal_reminder forces an actual choice.
    """
    siblings = [t for t in BY_DOMAIN[tool.domain] if t.name != tool.name]
    others = [t for t in TOOLS if t.domain != tool.domain]
    n_extra = rng.randint(min_extra, max_extra)
    picked = []
    if prefer_siblings:
        picked += rng.sample(siblings, min(len(siblings), n_extra))
    remaining = n_extra - len(picked)
    if remaining > 0:
        picked += rng.sample(others, min(len(others), remaining))
    shown = [tool] + picked
    rng.shuffle(shown)
    return [t.schema() for t in shown]


def build_agentic(n, seed):
    rng = random.Random(seed)
    counts = {k: int(n * v) for k, v in CATEGORY_MIX.items()}
    counts["simple"] += n - sum(counts.values())  # absorb rounding
    rows = []

    for _ in range(counts["simple"]):
        tool = rng.choice(TOOLS)
        rows.append({"slice": "agentic", "category": "simple",
                     "tools": visible_tools(tool, rng, 0, 1, prefer_siblings=False),
                     "messages": [{"role": "user",
                                   "content": frame(tool.render_ask(rng), rng)}]})

    for _ in range(counts["argtype"]):
        # bias towards tools that actually have typed or enumerated parameters
        typed = [t for t in TOOLS
                 if any(isinstance(s, (list, tuple)) or s in ("integer", "number", "boolean")
                        for s, _, _ in t.params.values())]
        tool = rng.choice(typed or TOOLS)
        ask = f"{tool.render_ask(rng)} {rng.choice(ARGTYPE_NUDGES)}"
        rows.append({"slice": "agentic", "category": "argtype",
                     "tools": visible_tools(tool, rng, 0, 2, prefer_siblings=False),
                     "messages": [{"role": "user", "content": ask}]})

    for _ in range(counts["selection"]):
        tool = rng.choice([t for t in TOOLS if len(BY_DOMAIN[t.domain]) > 1])
        rows.append({"slice": "agentic", "category": "selection",
                     "tools": visible_tools(tool, rng, 2, 5, prefer_siblings=True),
                     "messages": [{"role": "user",
                                   "content": frame(tool.render_ask(rng), rng)}]})

    for _ in range(counts["negative"]):
        # show real tools, then ask something none of them should be used for
        anchor = rng.choice(TOOLS)
        rows.append({"slice": "agentic", "category": "negative",
                     "tools": visible_tools(anchor, rng, 1, 3, prefer_siblings=True),
                     "messages": [{"role": "user", "content": render_negative(rng)}]})

    # Only tools whose own requests mention a slot a follow-up can vary. Anything else
    # produces an incoherent second turn, which is worse than no multi-turn data.
    multiturn_pool = tools_with_follow_ups()
    acks = ["Done. Anything else?", "That's done.", "All set.",
            "Sorted. What else?", "Done -- anything further?"]
    made = 0
    while made < counts["multiturn"]:
        tool = rng.choice(multiturn_pool)
        pair = build_multiturn(tool, rng)
        if pair is None:
            continue  # this template had no varyable slot; draw again
        first, follow = pair
        rows.append({"slice": "agentic", "category": "multiturn",
                     "tools": visible_tools(tool, rng, 1, 3, prefer_siblings=True),
                     "messages": [
                         {"role": "user", "content": first},
                         {"role": "assistant", "content": rng.choice(acks)},
                         {"role": "user", "content": follow},
                     ]})
        made += 1

    rng.shuffle(rows)
    print(f"  hand-written: {dict(counts)} across {len(TOOLS)} tools in "
          f"{len(BY_DOMAIN)} domains")
    report_diversity(rows, "hand-written")
    return rows


def build_toolace(n, seed, path=None):
    """Sample prompts from ToolACE, for schema breadth the hand-written catalog cannot give.

    Its schemas are RapidAPI-flavoured -- sports, finance, crypto -- which is a different
    distribution from the benchmark's everyday tools. That is why it supplements the
    hand-written catalog rather than replacing it: one keeps the domain distribution
    close to the eval, the other stops the model memorising a closed tool set.
    """
    from toolace import load_prompts, schema_stats

    rows, stats = load_prompts(path=path)
    print(f"  toolace: parsed {stats['kept']} rows "
          f"({stats['unparsed']} unparsable, {stats['no_usable_tools']} with no usable tools)")
    rng = random.Random(seed + 1)
    if n < len(rows):
        rows = rng.sample(rows, n)
    elif n > len(rows):
        print(f"  toolace: only {len(rows)} rows available, {n} requested; using all")
    print(f"  toolace: {schema_stats(rows)}")
    report_diversity(rows, "toolace")
    return rows


def report_diversity(rows, label="agentic", warn=False):
    """Duplicates are weak training signal. Say plainly how many there are.

    Two numbers matter and they are not the same thing. Distinct PROMPTS says whether
    the text repeats. Impressions per SCHEMA says whether the model could memorise the
    tool set instead of learning to read a function definition -- which is what the
    benchmark, scored on seven unseen schemas, actually tests.
    """
    seen = [json.dumps(r["messages"], sort_keys=True) for r in rows]
    unique = len(set(seen))
    pct = 100 * unique / len(rows) if rows else 0
    names = {t["function"]["name"] for r in rows for t in r.get("tools") or []}
    impressions = sum(len(r.get("tools") or []) for r in rows)
    per = impressions / max(len(names), 1)
    print(f"  {label}: distinct prompts {unique}/{len(rows)} ({pct:.1f}%)")
    print(f"  {label}: {len(names)} distinct schemas, {per:.1f} impressions each")
    # Only judge the pool that actually gets written. A sub-slice showing its schemas
    # often is fine and expected: the hand-written catalog exists to match the
    # benchmark's domains, not to supply breadth.
    if not warn:
        return
    if pct < 80:
        print("  WARNING: heavy prompt repetition. Widen tools_train.py or lower --total.")
    if per > 60:
        print("  WARNING: each schema is shown very often. The model can memorise this "
              "tool set rather than learn to read schemas. Raise --toolace-frac.")


def build_reasoning(n, seed):
    from datasets import load_dataset

    banned = gsm8k_eval_questions()
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows, skipped = [], 0
    for i in idx:
        q = ds[i]["question"].strip()
        if q in banned:
            skipped += 1
            continue
        rows.append({"slice": "reasoning",
                     "messages": [{"role": "user", "content": q}]})
        if len(rows) >= n:
            break
    print(f"  reasoning: {len(rows)} kept, {skipped} skipped as benchmark overlap")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=20000, help="total prompts across all slices")
    ap.add_argument("--general-dataset", default="HuggingFaceFW/fineweb-edu",
                    help="HF dataset id, or a path to a local jsonl with a text field")
    ap.add_argument("--general-split", default="train")
    ap.add_argument("--general-field", default="text")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=["general", "agentic", "reasoning"])
    ap.add_argument("--toolace-frac", type=float, default=0.6,
                    help="share of the agentic slice drawn from ToolACE rather than "
                         "the hand-written catalog. 0 disables ToolACE entirely.")
    ap.add_argument("--toolace-path", default=None,
                    help="local data.json; downloads from Hugging Face if omitted")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Hard guard: the training catalog must never reuse a benchmark tool name, or the
    # tools_acc number stops measuring generalisation and starts measuring memorisation.
    overlap = {t.name for t in TOOLS} & EVAL_TOOL_NAMES
    if overlap:
        raise SystemExit(f"training catalog collides with benchmark tools: {sorted(overlap)}")

    counts = {k: int(args.total * v) for k, v in MIX.items()}
    print(f"target mix: {counts}")

    if "general" not in args.skip:
        rows = build_general(counts["general"], args.general_dataset,
                             args.general_split, args.general_field, args.seed)
        write_jsonl(out / "prompts-general.jsonl", rows)
        print(f"wrote {len(rows)} -> {out / 'prompts-general.jsonl'}")

    generate_rows = []
    if "agentic" not in args.skip:
        n_toolace = int(counts["agentic"] * args.toolace_frac)
        n_hand = counts["agentic"] - n_toolace
        generate_rows += build_agentic(n_hand, args.seed)
        if n_toolace:
            generate_rows += build_toolace(n_toolace, args.seed, args.toolace_path)
    if "reasoning" not in args.skip:
        generate_rows += build_reasoning(counts["reasoning"], args.seed)
    if generate_rows:
        random.Random(args.seed).shuffle(generate_rows)
        # Final guard across BOTH sources. toolace.py filters colliding schemas as it
        # parses, but the assertion belongs where the file is written, not where one
        # producer happens to run.
        leaked = {t["function"]["name"] for r in generate_rows
                  for t in r.get("tools") or []} & EVAL_TOOL_NAMES
        if leaked:
            raise SystemExit(f"benchmark tool schemas leaked into training data: {sorted(leaked)}")
        write_jsonl(out / "prompts-generate.jsonl", generate_rows)
        print(f"\nwrote {len(generate_rows)} -> {out / 'prompts-generate.jsonl'}")
        report_diversity(generate_rows, "combined", warn=True)

    print("\nnext:")
    print(f"  gen_teacher.py --mode forward  --input {out / 'prompts-general.jsonl'}  --out data/general")
    print(f"  gen_teacher.py --mode generate --input {out / 'prompts-generate.jsonl'} --out data/agentic")


if __name__ == "__main__":
    main()
