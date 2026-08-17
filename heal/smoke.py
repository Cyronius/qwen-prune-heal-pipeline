"""Run the whole healing pipeline on CPU, at toy scale, in about a minute.

The point is to catch plumbing bugs here rather than on a rented GPU billed by the
hour. It builds a randomly-initialised model with the same architecture and the same
layer pattern as the real pruned checkpoint, only tiny, then runs generation,
training and merging against it end to end.

The vocabulary and tokenizer are the REAL ones, so the chat template, the tool
serialisation and the top-k shard sizes all exercise real code paths.

Usage:
  python smoke.py --reference ../pruned36-c-only --work <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from common import tiny_config  # noqa: E402


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(str(c) for c in cmd)}")


def build_tiny_checkpoint(reference: Path, dst: Path, vocab_size: int):
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

    cfg = tiny_config(reference)
    cfg.text_config.vocab_size = vocab_size
    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(cfg).to(torch.float32)
    n = sum(p.numel() for p in model.parameters())
    model.save_pretrained(str(dst), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(reference)).save_pretrained(str(dst))
    for extra in ("chat_template.jinja",):
        if (reference / extra).exists():
            (dst / extra).write_text((reference / extra).read_text(encoding="utf-8"),
                                     encoding="utf-8")
    print(f"tiny checkpoint: {n/1e6:.1f}M params -> {dst}")
    return cfg


def inject_fake_mtp(dst: Path, hidden: int):
    """Give the toy checkpoint mtp.* tensors, so merge_heal's reattach path is tested.

    transformers ignores these on load, which is exactly the behaviour we want to
    prove we work around.
    """
    idx_path = dst / "model.safetensors.index.json"
    tensors = {
        "mtp.fc.weight": torch.zeros(hidden, 2 * hidden),
        "mtp.norm.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(hidden),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(hidden),
    }
    shard = "model-mtp-source.safetensors"
    save_file(tensors, dst / shard, metadata={"format": "pt"})
    if idx_path.exists():
        index = json.loads(idx_path.read_text(encoding="utf-8"))
    else:
        single = dst / "model.safetensors"
        weight_map = {}
        with safe_open(single, framework="pt") as f:
            for k in f.keys():
                weight_map[k] = "model.safetensors"
        index = {"metadata": {}, "weight_map": weight_map}
    for name in tensors:
        index["weight_map"][name] = shard
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"injected {len(tensors)} fake mtp.* tensors")
    return set(tensors)


def make_prompts(work: Path, catalog_path: Path):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    tool = next(iter(catalog.values()), None)

    general = [{"slice": "general", "text": t} for t in [
        "The capital of France is Paris, a city known for its museums and its river.",
        "Photosynthesis converts light energy into chemical energy stored in sugars.",
        "A prime number has exactly two distinct positive divisors, one and itself.",
        "The Pacific Ocean is the largest and deepest of the world ocean basins.",
    ]]
    agentic = []
    for city in ["Tokyo", "Oslo", "Lima"]:
        row = {"slice": "agentic",
               "messages": [{"role": "user", "content": f"What is the weather in {city}?"}]}
        if tool:
            row["tools"] = [tool]
        agentic.append(row)
    agentic.append({"slice": "agentic",
                    "messages": [{"role": "user", "content": "Who wrote Hamlet?"}],
                    "tools": [tool] if tool else None})

    (work / "prompts-general.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in general), encoding="utf-8")
    (work / "prompts-agentic.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in agentic), encoding="utf-8")
    print(f"wrote {len(general)} general and {len(agentic)} agentic prompts")


def check_toolace():
    """Validate the ToolACE ingestion, but only if the file is already cached.

    The smoke test must run offline. A missing cache is reported and skipped rather
    than triggering a 37 MB download in the middle of a two-minute check.
    """
    from tools_train import EVAL_TOOL_NAMES

    try:
        from huggingface_hub import hf_hub_download

        from toolace import REPO, FILENAME
        path = hf_hub_download(REPO, FILENAME, repo_type="dataset",
                               local_files_only=True)
    except Exception:
        print("  SKIP: ToolACE not cached. Run `python toolace.py` once to fetch it.")
        return

    from toolace import load_prompts, schema_stats, slug

    if slug("Market Trends API") != "market_trends_api":
        raise SystemExit("FAILED: name slugging is wrong")

    rows, stats = load_prompts(path=path, limit=800)
    if not rows:
        raise SystemExit("FAILED: ToolACE parsed to zero usable rows")

    leaked = {t["function"]["name"] for r in rows for t in r["tools"]} & EVAL_TOOL_NAMES
    if leaked:
        raise SystemExit(f"FAILED: benchmark schemas survived the filter: {sorted(leaked)}")

    for r in rows:
        if not r["tools"]:
            raise SystemExit("FAILED: row with no tools")
        if r["messages"][-1]["role"] != "user":
            raise SystemExit("FAILED: row does not end on a user turn")
        for t in r["tools"]:
            fn = t["function"]
            if not re.fullmatch(r"[a-z0-9_]+", fn["name"]):
                raise SystemExit(f"FAILED: illegal function name {fn['name']!r}")
            if fn["parameters"].get("type") != "object":
                raise SystemExit(f"FAILED: {fn['name']} params not typed object")
            json.dumps(t)

    st = schema_stats(rows)
    print(f"  OK: {st['rows']} rows, {st['distinct_schemas']} schemas, "
          f"{st['impressions_per_schema']} impressions each, no benchmark leakage")


def check_collate_alignment():
    """Prove the teacher's distributions line up with the right student positions.

    This is the one bug in the pipeline that cannot be caught by watching the loss.
    Misaligned KD targets still produce a smooth, falling curve; they just teach the
    model the wrong thing. So test it directly.

    The trick: build rows whose top-1 teacher token at every recorded step IS the token
    that actually follows. After collate, idx[..., 0] must equal the next token at every
    position the kd_mask selects. If the offset is wrong, it will not.
    """
    from train_heal import collate

    cases = [
        ("forward mode, teacher records every position", 12, 0),
        ("generate mode, teacher records only the sampled tail", 12, 7),
        ("generate mode, prompt of one token", 6, 1),
    ]
    for label, n_tokens, n_prompt in cases:
        tokens = torch.arange(100, 100 + n_tokens)
        start = max(n_prompt - 1, 0)
        m = (n_tokens - 1) - start
        topk_idx = torch.zeros((m, 64), dtype=torch.long)
        # top-1 is the true next token for each recorded step
        topk_idx[:, 0] = tokens[start + 1: n_tokens]
        row = {"tokens": tokens, "topk_idx": topk_idx,
               "topk_logprob": torch.zeros((m, 64)), "start": start}
        batch = collate([row], pad_id=0)
        mask = batch["kd_mask"][0]
        if int(mask.sum()) != m:
            raise SystemExit(f"FAILED [{label}]: kd_mask covers {int(mask.sum())}, expected {m}")
        pos = mask.nonzero().flatten()
        got = batch["topk_idx"][0, pos, 0]
        want = batch["input_ids"][0, pos + 1]
        if not torch.equal(got, want):
            raise SystemExit(
                f"FAILED [{label}]: KD targets misaligned\n  got  {got.tolist()}\n  want {want.tolist()}")
        print(f"  OK: {label}")


def check_tool_catalog():
    """Validate the training catalog before anything expensive touches it."""
    import random
    import re

    from tools_train import (BY_DOMAIN, EVAL_TOOL_NAMES, NEGATIVE_TEMPLATES, TOOLS,
                             build_multiturn, render_negative, tools_with_follow_ups)

    rng = random.Random(0)

    overlap = {t.name for t in TOOLS} & EVAL_TOOL_NAMES
    if overlap:
        raise SystemExit(f"FAILED: training catalog reuses benchmark tools: {sorted(overlap)}")
    print(f"  OK: no overlap with the {len(EVAL_TOOL_NAMES)} benchmark tools")

    names = [t.name for t in TOOLS]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"FAILED: duplicate tool names: {dupes}")

    thin = [d for d, tools in BY_DOMAIN.items() if len(tools) < 2]
    if thin:
        raise SystemExit(f"FAILED: domains with no confusable sibling: {thin}")
    print(f"  OK: {len(TOOLS)} tools, {len(BY_DOMAIN)} domains, every domain has siblings")

    # Every request template must fill completely, and every schema must serialise.
    leftover = re.compile(r"\{\w+\}")
    for tool in TOOLS:
        json.dumps(tool.schema())
        for _ in range(20):
            text = tool.render_ask(rng)
            if leftover.search(text):
                raise SystemExit(f"FAILED: {tool.name} left a placeholder unfilled: {text}")
    for _ in range(300):
        text = render_negative(rng)
        if leftover.search(text):
            raise SystemExit(f"FAILED: negative left a placeholder unfilled: {text}")
    print(f"  OK: all templates fill; {len(NEGATIVE_TEMPLATES)} negative shapes")

    # A follow-up must change a value the first turn actually stated.
    pool = tools_with_follow_ups()
    if not pool:
        raise SystemExit("FAILED: no tool can produce a multi-turn exchange")
    checked = 0
    for tool in pool:
        for _ in range(5):
            pair = build_multiturn(tool, rng)
            if pair is None:
                continue
            first, follow = pair
            varied = leftover.sub("", follow)
            if follow.strip() and follow in first:
                raise SystemExit(
                    f"FAILED: {tool.name} follow-up restates the first turn\n"
                    f"  first  {first}\n  follow {follow}")
            checked += 1
    print(f"  OK: {checked} multi-turn pairs coherent across {len(pool)} tools")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(HERE.parent / "pruned36-c-only"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true", help="do not wipe the work dir first")
    args = ap.parse_args()

    reference = Path(args.reference)
    work = Path(args.work)
    if work.exists() and not args.keep:
        import shutil as sh
        sh.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(reference))
    vocab = len(tok)
    print(f"real tokenizer vocab: {vocab}")

    step("0a/6 training tool catalog (pure python, no model)")
    check_tool_catalog()

    step("0b/6 ToolACE ingestion (offline; skipped if not cached)")
    check_toolace()

    step("0c/6 KD target alignment (pure python, no model)")
    check_collate_alignment()

    step("1/6 build a tiny checkpoint with the real tokenizer")
    cfg = build_tiny_checkpoint(reference, work / "tiny", vocab)
    mtp_names = inject_fake_mtp(work / "tiny", cfg.text_config.hidden_size)

    step("2/6 write toy prompts")
    make_prompts(work, HERE.parent / "bench" / "tools_catalog.json")

    step("3/6 teacher forward pass over general text")
    run(["gen_teacher.py", "--teacher", work / "tiny", "--mode", "forward",
         "--input", work / "prompts-general.jsonl", "--out", work / "data-general",
         "--max-len", "64", "--log-every", "2", "--experts-impl", ""])

    step("4/6 teacher generation over agentic prompts")
    run(["gen_teacher.py", "--teacher", work / "tiny", "--mode", "generate",
         "--input", work / "prompts-agentic.jsonl", "--out", work / "data-agentic",
         "--max-new", "8", "--temperature", "0.7", "--log-every", "1",
         "--experts-impl", ""])

    step("5/6 train, both losses")
    # max-len must exceed the agentic prompts, which carry serialised tool schemas.
    # Anything shorter truncates them to zero trainable tokens and the loss goes NaN.
    for loss in ("sft", "kd"):
        run(["train_heal.py", "--student", work / "tiny",
             "--data", work / "data-general", work / "data-agentic",
             "--out", work / f"run-{loss}", "--loss", loss,
             "--max-len", "1024", "--steps", "4", "--grad-accum", "2",
             "--eval-every", "2", "--save-every", "4", "--warmup", "2",
             "--rank", "4", "--alpha", "8", "--device", "cpu",
             "--experts-impl", ""])
        log = work / f"run-{loss}" / "log.jsonl"
        evals = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()
                 if l.strip() and "eval_ce" in l]
        if not evals:
            raise SystemExit(f"FAILED: {loss} run produced no eval entries")
        bad = [e for e in evals if e["eval_ce"] != e["eval_ce"]]  # NaN check
        if bad:
            raise SystemExit(f"FAILED: {loss} run produced NaN eval loss: {bad}")
        print(f"  {loss}: eval ce {evals[-1]['eval_ce']:.4f} (finite, good)")

    step("6/6 merge and check the mtp tensors survived")
    run(["merge_heal.py", "--student", work / "tiny",
         "--adapter", work / "run-sft" / "adapter-final",
         "--out", work / "merged", "--device-map", "cpu", "--experts-impl", ""])

    merged_index = json.loads(
        (work / "merged" / "model.safetensors.index.json").read_text(encoding="utf-8"))
    missing = mtp_names - set(merged_index["weight_map"])
    if missing:
        raise SystemExit(f"FAILED: mtp tensors lost in merge: {sorted(missing)}")
    print(f"\nOK: all {len(mtp_names)} mtp.* tensors present in the merged checkpoint")
    print("smoke test passed")


if __name__ == "__main__":
    main()
