"""Heal a pruned checkpoint with LoRA.

Two losses, selected by --loss.

  sft   Cross-entropy against the teacher's sampled tokens. Simple. Run this first.
  kd    KL divergence against the teacher's stored top-k distribution, blended with
        the sft loss. Better signal per token, more moving parts. Run this only if
        sft undershoots the ship criteria.

Only LoRA adapters train. The routed experts, the router, the embeddings, the lm_head
and the vision tower all stay frozen. See common.LORA_TARGET_REGEX for why.

Usage:
  python train_heal.py --student pruned36-c-only --data data/general data/agentic \
      --out runs/heal-c --loss sft --four-bit
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import TOP_K, build_lora, freeze_vision, load_student, read_jsonl


class ShardDataset:
    """Reads the .npz shards written by gen_teacher.py, one shard resident at a time."""

    def __init__(self, dirs, max_len):
        self.max_len = max_len
        rows = []
        for d in dirs:
            d = Path(d)
            for r in read_jsonl(d / "index.jsonl"):
                r["dir"] = str(d)
                rows.append(r)
        # A sequence whose prompt runs past max_len has no supervised positions left
        # after truncation. Its cross-entropy divides by zero and returns NaN, which
        # then poisons every gradient in the accumulation group. Drop them here.
        self.rows = [r for r in rows if self._n_labels(r) > 0]
        dropped = len(rows) - len(self.rows)
        if dropped:
            print(f"dropped {dropped}/{len(rows)} sequences: prompt longer than "
                  f"--max-len {max_len}, so no tokens left to train on")
        if not self.rows:
            raise SystemExit("no usable sequences; raise --max-len")
        self._cache_key, self._cache = None, None

    def _n_labels(self, row):
        t = min(int(row["n_tokens"]), self.max_len)
        start = max(int(row.get("n_prompt_tokens", 0)) - 1, 0)
        return max(t - (start + 1), 0)

    def __len__(self):
        return len(self.rows)

    def _shard(self, row):
        key = (row["dir"], row["shard"])
        if key != self._cache_key:
            self._cache = np.load(Path(row["dir"]) / f"shard-{row['shard']:05d}.npz")
            self._cache_key = key
        return self._cache

    def __getitem__(self, i):
        row = self.rows[i]
        z = self._shard(row)
        k = row["key"]
        tokens = torch.from_numpy(z[f"{k}.tokens"].astype(np.int64))[: self.max_len]
        idx = torch.from_numpy(z[f"{k}.topk_idx"].astype(np.int64))
        lp = torch.from_numpy(z[f"{k}.topk_logprob"].astype(np.float32))
        # start is where the teacher's recorded distributions begin, in shifted-logit
        # coordinates. forward mode records every position, so start is 0. generate
        # mode records only the sampled steps, and the distribution that produced the
        # first generated token sits at shifted index n_prompt - 1.
        start = max(int(row.get("n_prompt_tokens", 0)) - 1, 0)
        return {"tokens": tokens, "topk_idx": idx, "topk_logprob": lp, "start": start}


def collate(batch, pad_id):
    n = max(len(b["tokens"]) for b in batch)
    tokens = torch.full((len(batch), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    idx = torch.zeros((len(batch), n - 1, TOP_K), dtype=torch.long)
    lp = torch.full((len(batch), n - 1, TOP_K), -1e4, dtype=torch.float32)
    kd_mask = torch.zeros((len(batch), n - 1), dtype=torch.bool)
    for i, b in enumerate(batch):
        t = len(b["tokens"])
        tokens[i, :t] = b["tokens"]
        attn[i, :t] = 1
        s = b["start"]
        labels[i, s + 1: t] = b["tokens"][s + 1: t]
        # The teacher's arrays start at shifted index s, not at 0. For generate mode
        # that is the prompt boundary. Writing them at 0 would pair every teacher
        # distribution with the wrong student position, and the KD loss would train
        # against noise while still looking like it was converging.
        m = min(len(b["topk_idx"]), (n - 1) - s)
        if m > 0:
            idx[i, s: s + m] = b["topk_idx"][:m]
            lp[i, s: s + m] = b["topk_logprob"][:m]
            kd_mask[i, s: s + m] = True
    return {"input_ids": tokens, "attention_mask": attn, "labels": labels,
            "topk_idx": idx, "topk_logprob": lp, "kd_mask": kd_mask}


def kd_loss(student_logits, topk_idx, topk_logprob, kd_mask):
    """Sparse KL over the teacher's top-k support.

    Both sides are renormalised over the same k tokens, so the divergence is measured
    on the support the teacher actually recorded rather than on a truncated full-vocab
    distribution that would not sum to one.
    """
    sl = student_logits[:, :-1, :]
    sel = torch.gather(sl, -1, topk_idx)                       # [B, T-1, k]
    student_lp = torch.log_softmax(sel.float(), dim=-1)
    teacher_lp = torch.log_softmax(topk_logprob.float(), dim=-1)
    kl = (teacher_lp.exp() * (teacher_lp - student_lp)).sum(-1)  # [B, T-1]
    denom = kd_mask.sum().clamp(min=1)
    return (kl * kd_mask).sum() / denom


@torch.no_grad()
def evaluate(model, loader_batches, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader_batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"] != -100).sum())
        if n == 0:
            continue
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        total += float(out.loss.detach()) * n
        count += n
    model.train()
    if count == 0:
        return float("nan"), float("nan")
    ce = total / count
    return ce, math.exp(min(ce, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", choices=["sft", "kd"], default="sft")
    ap.add_argument("--kd-weight", type=float, default=0.7)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-frac", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--experts-impl", default="grouped_mm")
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--text-only", action="store_true",
                    help="drop the vision tower instead of carrying it frozen")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    cpu = args.device == "cpu" or not torch.cuda.is_available()
    model = load_student(args.student, four_bit=args.four_bit,
                         dtype=torch.float32 if cpu else torch.bfloat16,
                         experts_impl=args.experts_impl, attn_impl=args.attn_impl,
                         text_only=args.text_only,
                         device_map=None if cpu else (
                             {"": args.device} if args.device != "auto" else "auto"))
    if cpu:
        model = model.to("cpu")
    model.config.use_cache = False
    if args.four_bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    frozen = freeze_vision(model)
    model = build_lora(model, rank=args.rank, alpha=args.alpha)
    model.print_trainable_parameters()
    print(f"froze {frozen} vision parameters")

    ds = ShardDataset(args.data, args.max_len)
    order = list(range(len(ds)))
    random.shuffle(order)
    n_eval = max(1, int(len(order) * args.eval_frac))
    eval_ids, train_ids = order[:n_eval], order[n_eval:]
    print(f"{len(train_ids)} train sequences, {len(eval_ids)} eval sequences")

    eval_batches = [collate([ds[i]], pad_id) for i in eval_ids]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)))

    device = "cpu" if cpu else (args.device if args.device != "auto" else "cuda")
    cursor, t0, tokens_seen = 0, time.time(), 0
    log = []
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            picks = []
            for _ in range(args.batch_size):
                if cursor >= len(train_ids):
                    random.shuffle(train_ids)
                    cursor = 0
                picks.append(ds[train_ids[cursor]])
                cursor += 1
            batch = {k: v.to(device) for k, v in collate(picks, pad_id).items()}
            if int((batch["labels"] != -100).sum()) == 0:
                continue  # belt and braces; ShardDataset already filters these out
            out_ = model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"],
                         labels=batch["labels"])
            loss = out_.loss
            if args.loss == "kd":
                kl = kd_loss(out_.logits, batch["topk_idx"], batch["topk_logprob"],
                             batch["kd_mask"])
                loss = args.kd_weight * kl + (1 - args.kd_weight) * loss
            if not torch.isfinite(loss):
                print(f"  step {step}: non-finite loss, skipping microbatch", flush=True)
                continue
            (loss / args.grad_accum).backward()
            step_loss += float(loss.detach()) / args.grad_accum
            tokens_seen += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % 10 == 0:
            tps = tokens_seen / max(time.time() - t0, 1e-9)
            print(f"step {step:5d}  loss {step_loss:.4f}  {tps:8.1f} tok/s", flush=True)
            log.append({"step": step, "loss": step_loss, "tok_per_s": tps})
        if step % args.eval_every == 0:
            ce, ppl = evaluate(model, eval_batches, device)
            print(f"  eval  ce {ce:.4f}  ppl {ppl:.3f}", flush=True)
            log.append({"step": step, "eval_ce": ce, "eval_ppl": ppl})
            (out / "log.jsonl").write_text(
                "\n".join(json.dumps(r) for r in log), encoding="utf-8")
        if step % args.save_every == 0:
            model.save_pretrained(out / f"adapter-step{step}")

    model.save_pretrained(out / "adapter-final")
    (out / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
    print(f"saved to {out / 'adapter-final'}")


if __name__ == "__main__":
    main()
