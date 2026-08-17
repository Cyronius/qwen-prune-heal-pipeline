"""Build the healing dataset by recording what the teacher does.

Two modes, because the two data slices need different things.

  forward   Teacher-force over text that already exists. One forward pass gives a
            distribution at every position. Cheap per token. Used for the general
            web-text slice, whose job is to restore perplexity and knowledge.

  generate  Sample a completion from the teacher, recording the distribution at each
            step. Expensive per token. Used for the agentic and reasoning slices,
            where the teacher's own behaviour is the thing being copied.

Note on llama-server: its /completion endpoint returns probabilities only for tokens
it generated, never for prompt tokens. So it cannot do forward mode at all. That is
why this script uses transformers for both modes rather than splitting backends.

Output is a jsonl index plus .npz shards of top-k log-probabilities.

Usage:
  python gen_teacher.py --teacher <path> --mode forward  --input prompts.jsonl --out data/general
  python gen_teacher.py --teacher <path> --mode generate --input prompts.jsonl --out data/agentic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import TOP_K, read_jsonl, student_class


def topk_from_logits(logits, k):
    """logits: [T, V] -> (int32 indices [T,k], float16 log-probs [T,k])."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(logprobs, k, dim=-1)
    return idx.to(torch.int32).cpu().numpy(), vals.to(torch.float16).cpu().numpy()


class ShardWriter:
    """Rolling .npz shards, so a crash costs one shard and not the whole run."""

    def __init__(self, out_dir, shard_mb=512):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.limit = shard_mb * 1024 * 1024
        self.buf, self.bytes, self.shard = {}, 0, 0
        self.index = []

    def add(self, key, tokens, idx, logprobs, meta):
        self.buf[f"{key}.tokens"] = tokens
        self.buf[f"{key}.topk_idx"] = idx
        self.buf[f"{key}.topk_logprob"] = logprobs
        self.bytes += tokens.nbytes + idx.nbytes + logprobs.nbytes
        self.index.append({**meta, "key": key, "shard": self.shard, "n_tokens": int(len(tokens))})
        if self.bytes >= self.limit:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        np.savez(self.dir / f"shard-{self.shard:05d}.npz", **self.buf)
        self.buf, self.bytes = {}, 0
        self.shard += 1

    def close(self):
        self.flush()
        with (self.dir / "index.jsonl").open("w", encoding="utf-8") as f:
            for row in self.index:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(self.index), sum(r["n_tokens"] for r in self.index)


@torch.no_grad()
def run_forward(model, tok, rows, writer, max_len, log_every):
    """Teacher-force over provided text. Distribution at position i predicts token i+1."""
    t0 = time.time()
    for i, row in enumerate(rows):
        ids = tok(row["text"], return_tensors="pt", truncation=True, max_length=max_len)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        logits = model(**ids).logits[0]  # [T, V]
        tokens = ids["input_ids"][0].to(torch.int32).cpu().numpy()
        # drop the final position: it predicts a token we do not have
        idx, lp = topk_from_logits(logits[:-1], TOP_K)
        writer.add(f"fwd-{i:07d}", tokens, idx, lp,
                   {"mode": "forward", "slice": row.get("slice", "general"),
                    "n_prompt_tokens": 0})
        if log_every and i % log_every == 0:
            done = i + 1
            print(f"  forward {done}/{len(rows)}  {done / max(time.time() - t0, 1e-9):.2f} seq/s", flush=True)


def run_generate(model, tok, rows, writer, max_new, temperature, log_every):
    """Sample completions, recording the distribution at each generated step."""
    t0 = time.time()
    for i, row in enumerate(rows):
        chat = row.get("messages")
        if chat is not None:
            text = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                tools=row.get("tools"))
        else:
            text = row["text"]
        enc = tok(text, return_tensors="pt").to(model.device)
        n_prompt = enc["input_ids"].shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature or None,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seq = out.sequences[0]
        # scores[t] is the distribution that produced sequence position n_prompt + t
        scores = torch.stack(out.scores, dim=0)[:, 0, :]  # [new, V]
        idx, lp = topk_from_logits(scores, TOP_K)
        writer.add(f"gen-{i:07d}", seq.to(torch.int32).cpu().numpy(), idx, lp,
                   {"mode": "generate", "slice": row.get("slice", "agentic"),
                    "n_prompt_tokens": int(n_prompt)})
        if log_every and i % log_every == 0:
            done = i + 1
            print(f"  generate {done}/{len(rows)}  {done / max(time.time() - t0, 1e-9):.2f} seq/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="path to the unpruned bf16 model")
    ap.add_argument("--mode", choices=["forward", "generate"], required=True)
    ap.add_argument("--input", required=True, help="jsonl of prompts")
    ap.add_argument("--out", required=True, help="output directory for shards")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-mb", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--experts-impl", default="grouped_mm",
                    help="pass an empty string to use the slow default (CPU smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--text-only", action="store_true",
                    help="load via Qwen3_5MoeForCausalLM, dropping the vision tower")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} prompts from {args.input}")

    tok = AutoTokenizer.from_pretrained(args.teacher)
    cpu = args.device == "cpu" or not torch.cuda.is_available()
    kwargs = {"dtype": torch.float32 if cpu else torch.bfloat16}
    if not cpu:
        kwargs["device_map"] = args.device
    if args.experts_impl:
        kwargs["experts_implementation"] = args.experts_impl
    model = student_class(args.text_only).from_pretrained(args.teacher, **kwargs)
    model.eval()

    writer = ShardWriter(args.out, args.shard_mb)
    if args.mode == "forward":
        run_forward(model, tok, rows, writer, args.max_len, args.log_every)
    else:
        run_generate(model, tok, rows, writer, args.max_new, args.temperature, args.log_every)
    n_seq, n_tok = writer.close()
    print(f"wrote {n_seq} sequences / {n_tok} tokens to {args.out}")


if __name__ == "__main__":
    main()
