"""Merge GDN layer pairs instead of dropping one (Qwen3.5/3.6 MoE, e.g. 30 -> 20 layers).

The plain second cut (C2: drop GDN slot 1 of every [GDN, GDN, attn] group) killed the
healed 30-layer model: ppl 3,741, tools .05. This script makes the same cut, to the same
layer count and layout, but folds the dropped layer into its neighbour instead of
throwing it away. Calibration data picks the merge for each pair. Nothing here trains
by gradient; healing comes after, with the usual heal/ pipeline.

What one merged layer is made of
--------------------------------
Each decoder layer is  x -> x + T(norm1(x)) -> y + F(norm2(y)),  where T is the token
mixer (GDN) and F is the MoE. A pair (a, b) does T_a, F_a, T_b, F_b. One merged layer
gets one T and one F, so:

  mixer   a, b or avg. You cannot stack two GDNs into one, and their heads do not line
          up (head 3 of layer a has nothing to do with head 3 of layer b), so averaging
          them is expected to fail. It is in the menu so the numbers say so.

  moe     union: both layers' experts go into one pool (512), the 256 that carry the
          most routed output on calibration data stay. Experts are whole functions
          from the residual stream back into it, so no neuron matching is needed.
          Each layer's post-attention RMSNorm scale is folded into its own experts'
          and router rows' input columns, so they can share one norm (set to scale 1).
          The two shared experts are concatenated (512 + 512 -> 1024 wide), which is
          an exact sum except that they now share one gate.
          a / b: keep one layer's MoE. avg: naive average of everything by index.

  fit     full: a ReplaceMe-style linear map, fit by ridge least squares, that maps
          [routed output, shared output] onto what the pair actually added to the
          residual stream. It folds exactly into every expert's down_proj and the
          shared expert's down_proj, so the result is a plain checkpoint. This is the
          piece that puts back the update size lost when 16 expert slots become 8.
          diag: same, one scale per channel. none: no correction.

A variant is "mixer:moe:fit". Every variant is scored on held-out calibration sequences
by how far the merged layer's output lands from the original pair's output. The best one
per group is kept (or --force-variant picks one for every group).

Calibration is sequential: the merged model's own hidden states feed the next group, so
each fit also corrects drift from the groups before it. Three streams run side by side:
  teacher  the source model, unchanged
  merged   what this script writes
  drop     plain C2 (keep slot a, drop slot b), as the baseline
At the end all three get a held-out perplexity on the calibration text, so you see
merge vs. drop before converting or benchmarking anything.

Global side effects (both exact, both reported)
-----------------------------------------------
* With --shared concat, the config's shared_expert_intermediate_size doubles. Every
  layer that does not use a concatenated shared expert, and the MTP head, gets its
  shared expert zero-padded to the new width. Zero channels output exactly zero. The
  cost is about 3M wasted active params per padded layer (about 1% of active total);
  LoRA healing can put those channels to use, since LoRA trains the shared expert.
* --top-k changes routing for every layer, the copied ones included.

Usage
-----
  python merge_layers_qwen35.py merged-heal-c merged-c2 \\
      --calib heal-artifacts/calibration.txt

Streams one layer pair at a time. Peak RAM is roughly two source layers (~3.5 GB bf16)
plus three hidden-state streams (n_seq * seq_len * hidden * 4 bytes each; the default
64 x 512 tokens is ~270 MB per stream). CPU is fine; it is slow but nothing is huge.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_FMT = "model.language_model.layers.{}."
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
EMBED = "model.language_model.embed_tokens.weight"
FINAL_NORM = "model.language_model.norm.weight"
LM_HEAD = "lm_head.weight"
SHARD_BYTES = 2 * 1024**3

DEFAULT_VARIANTS = [
    "a:a:none",      # = plain drop of slot b (C2). The baseline.
    "b:b:none",      # = plain drop of slot a
    "avg:avg:none",  # naive parameter average of the two layers
    "a:a:full",      # keep layer a, refit its output map (ReplaceMe-style)
    "a:union:none",  # union experts, no correction (shows the halving problem)
    "a:union:full",
    "b:union:full",
    "avg:union:full",
]


# --------------------------------------------------------------------------------------
# checkpoint access
# --------------------------------------------------------------------------------------

class Checkpoint:
    """Lazy tensor reads from a sharded safetensors checkpoint."""

    def __init__(self, path: Path):
        self.path = path
        self.weight_map = json.load(open(path / "model.safetensors.index.json"))["weight_map"]

    def get(self, name):
        with safe_open(self.path / self.weight_map[name], framework="pt") as f:
            # .clone(): an mmap-backed tensor pins its whole shard (Windows commit blowup)
            return f.get_tensor(name).clone()

    def layer(self, i):
        """All tensors of layer i, keyed without the layer prefix.

        Routed experts always come back fused (mlp.experts.gate_up_proj [E, 2F, H],
        mlp.experts.down_proj [E, H, F]), whichever way the checkpoint stores them.
        Recent transformers saves them per expert (experts.N.gate_proj.weight ...).
        """
        prefix = LAYER_FMT.format(i)
        st = {n[len(prefix):]: self.get(n) for n in self.weight_map if n.startswith(prefix)}
        per_expert = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        split = {k: v for k, v in st.items() if per_expert.match(k)}
        if split:
            n = 1 + max(int(per_expert.match(k).group(1)) for k in split)
            st = {k: v for k, v in st.items() if k not in split}
            st["mlp.experts.gate_up_proj"] = torch.stack([
                torch.cat([split[f"mlp.experts.{e}.gate_proj.weight"],
                           split[f"mlp.experts.{e}.up_proj.weight"]]) for e in range(n)])
            st["mlp.experts.down_proj"] = torch.stack(
                [split[f"mlp.experts.{e}.down_proj.weight"] for e in range(n)])
        return st


class ShardWriter:
    def __init__(self, dst: Path):
        self.dst, self.buf, self.size, self.files, self.map = dst, {}, 0, [], {}

    def add(self, name, t):
        t = t.contiguous()
        self.buf[name] = t
        self.size += t.numel() * t.element_size()
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        fname = f"part-{len(self.files):05d}.safetensors"
        save_file(self.buf, self.dst / fname, metadata={"format": "pt"})
        self.map.update({k: fname for k in self.buf})
        self.files.append(fname)
        self.buf, self.size = {}, 0

    def finish(self):
        self.flush()
        total, final = len(self.files), {}
        for j, fname in enumerate(self.files):
            new = f"model-{j + 1:05d}-of-{total:05d}.safetensors"
            shutil.move(self.dst / fname, self.dst / new)
            final.update({k: new for k, v in self.map.items() if v == fname})
        nbytes = sum((self.dst / f).stat().st_size for f in set(final.values()))
        json.dump({"metadata": {"total_size": nbytes}, "weight_map": final},
                  open(self.dst / "model.safetensors.index.json", "w"), indent=2)
        return total, nbytes


# --------------------------------------------------------------------------------------
# layer pieces
# --------------------------------------------------------------------------------------

def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Mixer:
    """input_layernorm + token mixer (GDN or full attention), as real HF modules."""

    def __init__(self, cfg, idx, state):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeAttention, Qwen3_5MoeGatedDeltaNet)
        self.kind = cfg.layer_types[idx]
        self.eps = cfg.rms_norm_eps
        self.norm_w = state["input_layernorm.weight"].float()
        if self.kind == "linear_attention":
            key, cls = "linear_attn.", Qwen3_5MoeGatedDeltaNet
        else:
            key, cls = "self_attn.", Qwen3_5MoeAttention
        self.tensors = {k: v for k, v in state.items() if k.startswith(key)}
        self.tensors["input_layernorm.weight"] = state["input_layernorm.weight"]
        with torch.device("meta"):
            mod = cls(cfg, idx)
        mod.load_state_dict({k[len(key):]: v.float() for k, v in self.tensors.items()
                             if k.startswith(key)}, strict=True, assign=True)
        self.mod = mod.eval()

    def __call__(self, x, pos_emb):
        h = rms(x, self.eps) * (1.0 + self.norm_w)
        if self.kind == "linear_attention":
            return self.mod(hidden_states=h)
        return self.mod(hidden_states=h, position_embeddings=pos_emb, attention_mask=None)[0]


class MoE:
    """A routed + shared MoE that reads an un-scaled RMS-normalised input.

    Each layer's post-attention norm scale (1 + w) is kept per source and applied to
    that source's experts lazily, so experts from two layers can share one input.
    """

    def __init__(self, router, experts, shared, top_k, act):
        self.router = router      # [E, H] fp32, norm scale already folded in
        self.experts = experts    # list of (gate_up [E_s, 2F, H], down [E_s, H, F], idx, scale [H])
        self.shared = shared      # dict gate [F', H], up [F', H], down [H, F'], gate_vec [1, H]; fp32 folded
        self.top_k = top_k
        self.act = act

    @classmethod
    def from_layer(cls, state, top_k, act):
        s = 1.0 + state["post_attention_layernorm.weight"].float()
        gu, dn = state["mlp.experts.gate_up_proj"], state["mlp.experts.down_proj"]
        return cls(
            router=state["mlp.gate.weight"].float() * s,
            experts=[(gu, dn, e, s) for e in range(gu.shape[0])],
            shared={"gate": state["mlp.shared_expert.gate_proj.weight"].float() * s,
                    "up": state["mlp.shared_expert.up_proj.weight"].float() * s,
                    "down": state["mlp.shared_expert.down_proj.weight"].float(),
                    "gate_vec": state["mlp.shared_expert_gate.weight"].float() * s},
            top_k=top_k, act=act)

    def expert_weights(self, e):
        gu, dn, i, s = self.experts[e]
        return gu[i].float() * s, dn[i].float()

    def __call__(self, x, scores=None, chunk=8192):
        """x: [N, H] RMS-normalised. Returns routed [N, H], shared [N, H].

        scores, if given, is a [E] float64 tensor that collects each expert's routed
        contribution: sum over tokens of routing weight * ||expert output||.
        """
        routed = torch.zeros_like(x)
        shared = torch.empty_like(x)
        sh = self.shared
        for lo in range(0, x.shape[0], chunk):
            xc = x[lo:lo + chunk]
            probs = torch.softmax(xc @ self.router.T, dim=-1)
            topv, topi = probs.topk(self.top_k, dim=-1)
            topv = topv / topv.sum(-1, keepdim=True)
            for e in torch.unique(topi).tolist():
                tok, pos = torch.where(topi == e)
                gu, dn = self.expert_weights(e)
                g, u = (xc[tok] @ gu.T).chunk(2, dim=-1)
                out = (self.act(g) * u) @ dn.T
                w = topv[tok, pos]
                routed[lo:lo + chunk].index_add_(0, tok, out * w[:, None])
                if scores is not None:
                    scores[e] += (w * out.norm(dim=-1)).sum().double()
            s_out = (self.act(xc @ sh["gate"].T) * (xc @ sh["up"].T)) @ sh["down"].T
            shared[lo:lo + chunk] = torch.sigmoid(xc @ sh["gate_vec"].T) * s_out
        return routed, shared


def avg_state(sa, sb):
    return {k: ((sa[k].float() + sb[k].float()) / 2).to(sa[k].dtype) for k in sa}


def union_moe(ma: MoE, mb: MoE, score_a, score_b, n_keep, top_k, act, shared_mode):
    pool = torch.cat([score_a, score_b])
    keep = pool.topk(n_keep).indices.sort().values.tolist()
    experts = [(ma.experts + mb.experts)[i] for i in keep]
    router = torch.cat([ma.router, mb.router])[keep]
    if shared_mode == "concat":
        sa, sb = ma.shared, mb.shared
        shared = {"gate": torch.cat([sa["gate"], sb["gate"]]),
                  "up": torch.cat([sa["up"], sb["up"]]),
                  "down": torch.cat([sa["down"], sb["down"]], dim=1),
                  "gate_vec": (sa["gate_vec"] + sb["gate_vec"]) / 2}
    else:
        shared = dict(ma.shared)
    n_a = sum(1 for i in keep if i < len(score_a))
    kept_frac = (pool[keep].sum() / pool.sum()).item()
    return MoE(router, experts, shared, top_k, act), {"from_a": n_a, "from_b": n_keep - n_a,
                                                       "score_kept": round(kept_frac, 4)}


# --------------------------------------------------------------------------------------
# output-map fit (ReplaceMe-style, folded into down projections)
# --------------------------------------------------------------------------------------

def fit_map(routed, shared, target, mode, ridge):
    """Find maps A_r, A_s ([H, H]) so routed @ A_r + shared @ A_s ~= target.

    Ridge-regularised toward identity, so a weak signal leaves the layer alone.
    """
    H = routed.shape[1]
    if mode == "diag":
        # per channel j: target_j ~= alpha_j routed_j + beta_j shared_j
        r, s, t = routed.double(), shared.double(), target.double()
        rr, ss, rs = (r * r).sum(0), (s * s).sum(0), (r * s).sum(0)
        rt, st = (r * t).sum(0), (s * t).sum(0)
        lam = ridge * (rr + ss).mean()
        a11, a22, a12 = rr + lam, ss + lam, rs
        b1, b2 = rt + lam, st + lam
        det = a11 * a22 - a12 * a12
        alpha, beta = (b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det
        return torch.diag(alpha).float(), torch.diag(beta).float()
    X = torch.cat([routed, shared], dim=1).double()
    xtx = X.T @ X
    lam = ridge * torch.diagonal(xtx).mean()
    prior = torch.cat([torch.eye(H), torch.eye(H)]).double()
    rhs = X.T @ target.double() + lam * prior
    W = torch.linalg.solve(xtx + lam * torch.eye(2 * H, dtype=torch.float64), rhs)
    return W[:H].float(), W[H:].float()


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def load_calibration(src: Path, calib: Path, n_seq, seq_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src))
    ids = tok(open(calib, encoding="utf-8").read(), return_tensors="pt")["input_ids"][0]
    need = n_seq * seq_len
    if len(ids) < need:
        raise SystemExit(f"calibration text has {len(ids)} tokens, need {need}")
    return ids[:need].view(n_seq, seq_len)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path, help="source checkpoint (e.g. the healed 30-layer model)")
    p.add_argument("dst", type=Path)
    p.add_argument("--calib", type=Path, help="calibration text file")
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--heldout", type=int, default=8, help="sequences held out for scoring")
    p.add_argument("--group", type=int, default=3, help="layer-pattern period ([GDN,GDN,attn] = 3)")
    p.add_argument("--merge-slot", type=int, default=0, help="merge slots S and S+1 of each group")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--force-variant", default=None, help="use this variant for every group")
    p.add_argument("--top-k", type=int, default=None, help="experts per token (default: keep)")
    p.add_argument("--shared", choices=["concat", "first"], default="concat")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=4, help="sequences per mixer forward")
    p.add_argument("--dry-run", action="store_true", help="score and report, write nothing")
    return p.parse_args(argv)


def main(argv=None, calib_ids=None):
    args = parse_args(argv)
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    torch.manual_seed(0)
    src = Checkpoint(args.src)
    cfg_json = json.load(open(args.src / "config.json"))
    tcfg = AutoConfig.from_pretrained(str(args.src)).text_config
    tcfg._attn_implementation = "sdpa"  # mask=None + sdpa == causal
    act = ACT2FN[tcfg.hidden_act]
    eps = tcfg.rms_norm_eps
    n_layers, G, S = tcfg.num_hidden_layers, args.group, args.merge_slot
    types = tcfg.layer_types
    assert n_layers % G == 0, f"{n_layers} layers is not a whole number of {G}-groups"
    for g0 in range(0, n_layers, G):
        assert types[g0 + S] == types[g0 + S + 1] == "linear_attention", \
            f"slots {S},{S + 1} of group at {g0} are not both GDN: {types[g0:g0 + G]}"
    k_src = tcfg.num_experts_per_tok
    k_new = args.top_k or k_src
    n_exp = tcfg.num_experts
    sf = tcfg.shared_expert_intermediate_size
    new_sf = 2 * sf if args.shared == "concat" else sf
    variants = [args.force_variant] if args.force_variant else args.variants.split(",")
    for v in variants:
        m, e, f = v.split(":")
        assert m in ("a", "b", "avg") and e in ("a", "b", "avg", "union") and f in ("none", "diag", "full"), v

    ids = calib_ids if calib_ids is not None else load_calibration(
        args.src, args.calib, args.n_seq, args.seq_len)
    n_seq, L = ids.shape
    n_train = n_seq - args.heldout
    assert n_train > 0 and args.heldout > 0
    H = tcfg.hidden_size
    # position 0 of every sequence is an attention-sink token with outsized activations;
    # it would dominate every least-squares fit and error metric, so it is left out.
    tok_mask = torch.ones(n_seq, L, dtype=torch.bool)
    tok_mask[:, 0] = False
    train_m = tok_mask.clone(); train_m[n_train:] = False
    held_m = tok_mask.clone(); held_m[:n_train] = False
    train_m, held_m = train_m.view(-1), held_m.view(-1)

    rotary = Qwen3_5MoeTextRotaryEmbedding(tcfg)
    emb = src.get(EMBED).float()[ids]  # [n_seq, L, H]
    teacher, merged, drop = emb.clone(), emb.clone(), emb.clone()
    del emb

    def mix(mixer, x):
        out = torch.empty_like(x)
        for lo in range(0, x.shape[0], args.batch):
            xb = x[lo:lo + args.batch]
            pos = torch.arange(L).view(1, 1, -1).expand(3, xb.shape[0], -1)
            out[lo:lo + args.batch] = mixer(xb, rotary(xb, pos))
        return out

    def full_layer(mixer, moe, x, scores=None):
        """One unmodified decoder layer, on [n_seq, L, H]."""
        y = x + mix(mixer, x)
        r, s = moe(rms(y.view(-1, H), eps), scores)
        return y + (r + s).view_as(y)

    if not args.dry_run:
        args.dst.mkdir(parents=True, exist_ok=True)
        writer = ShardWriter(args.dst)

    def write_layer(new_idx, tensors):
        if args.dry_run:
            return
        pre = LAYER_FMT.format(new_idx)
        for k, v in tensors.items():
            writer.add(pre + k, v)

    def pad_shared(tensors, prefix="mlp.shared_expert."):
        g = tensors[prefix + "gate_proj.weight"]
        if g.shape[0] == new_sf:
            return
        extra = new_sf - g.shape[0]
        for k in ("gate_proj.weight", "up_proj.weight"):
            t = tensors[prefix + k]
            tensors[prefix + k] = torch.cat([t, t.new_zeros(extra, t.shape[1])])
        t = tensors[prefix + "down_proj.weight"]
        tensors[prefix + "down_proj.weight"] = torch.cat([t, t.new_zeros(t.shape[0], extra)], dim=1)

    report = {"src": str(args.src), "variants": variants, "top_k": k_new,
              "shared": args.shared, "ridge": args.ridge,
              "calib_tokens": int(ids.numel()), "groups": []}
    new_idx = 0
    t_start = time.time()
    for g0 in range(0, n_layers, G):
        for slot in range(G):
            i = g0 + slot
            if slot == S + 1:
                continue
            if slot != S:
                # ---- a layer that is copied unchanged ----
                st = src.layer(i)
                mixer = Mixer(tcfg, i, st)
                teacher = full_layer(mixer, MoE.from_layer(st, k_src, act), teacher)
                drop = full_layer(mixer, MoE.from_layer(st, k_src, act), drop)
                merged = full_layer(mixer, MoE.from_layer(st, k_new, act), merged)
                pad_shared(st)
                write_layer(new_idx, st)
                new_idx += 1
                print(f"[{time.time() - t_start:6.0f}s] layer {i} copied -> {new_idx - 1}", flush=True)
                continue

            # ---- the pair (a, b) -> one layer ----
            a, b = i, i + 1
            sa, sb = src.layer(a), src.layer(b)
            mix_a, mix_b = Mixer(tcfg, a, sa), Mixer(tcfg, b, sb)
            moe_a, moe_b = MoE.from_layer(sa, k_src, act), MoE.from_layer(sb, k_src, act)

            # teacher: the real pair, collecting expert importance on its natural routing
            score_a = torch.zeros(n_exp, dtype=torch.float64)
            score_b = torch.zeros(n_exp, dtype=torch.float64)
            t0 = teacher
            t1 = full_layer(mix_a, moe_a, t0, score_a)
            t2 = full_layer(mix_b, moe_b, t1, score_b)
            teacher = t2
            drop = full_layer(mix_a, moe_a, drop)  # C2 baseline: keep a, drop b

            # merged-stream candidates
            x0 = merged.view(-1, H)
            target = t2.view(-1, H)
            pair_update = (t2 - t0).view(-1, H)
            denom = pair_update[held_m].norm().item()
            mixers, mid_cache, union_info = {"a": mix_a, "b": mix_b}, {}, None
            if any(v.split(":")[0] == "avg" for v in variants):
                mixers["avg"] = Mixer(tcfg, a, avg_state(mix_a.tensors, mix_b.tensors))
            best = None
            rows = []
            for v in variants:
                mname, ename, fname = v.split(":")
                if mname not in mid_cache:
                    mid_cache[mname] = merged + mix(mixers[mname], merged)
                mid = mid_cache[mname].view(-1, H)
                if ename == "a":
                    moe = MoE.from_layer(sa, k_new, act)
                elif ename == "b":
                    moe = MoE.from_layer(sb, k_new, act)
                elif ename == "avg":
                    moe = MoE.from_layer(avg_state(sa, sb), k_new, act)
                else:
                    moe, union_info = union_moe(
                        MoE.from_layer(sa, k_new, act), MoE.from_layer(sb, k_new, act),
                        score_a, score_b, n_exp, k_new, act, args.shared)
                r, s = moe(rms(mid, eps))
                maps = None
                if fname != "none":
                    maps = fit_map(r[train_m], s[train_m], (target - mid)[train_m], fname, args.ridge)
                    r, s = r @ maps[0], s @ maps[1]
                out = mid + r + s
                err = (out[held_m] - target[held_m]).norm().item() / denom
                err_train = ((out[train_m] - target[train_m]).norm()
                             / (t2 - t0).view(-1, H)[train_m].norm()).item()
                cos = F.cosine_similarity(out[held_m] - x0[held_m],
                                          target[held_m] - x0[held_m], dim=-1).mean().item()
                rows.append({"variant": v, "heldout_rel_err": round(err, 4),
                             "train_rel_err": round(err_train, 4), "update_cos": round(cos, 4)})
                print(f"    group {g0 // G}  {v:16s} held-out rel err {err:.4f}  "
                      f"(train {err_train:.4f})  update cos {cos:.3f}", flush=True)
                if best is None or err < best[0]:
                    best = (err, v, out.view_as(merged), moe, maps)
                del r, s, out
            err, v, out, moe, maps = best
            merged = out
            del mid_cache
            print(f"[{time.time() - t_start:6.0f}s] layers {a}+{b} -> {new_idx} via {v}", flush=True)
            report["groups"].append({"layers": [a, b], "new_index": new_idx, "chosen": v,
                                     "union": union_info, "variants": rows})

            # materialise the chosen merged layer
            mname, ename, _ = v.split(":")
            mixer_t = mixers[mname].tensors
            out_t = {k: v_.clone() for k, v_ in mixer_t.items()}
            dtype = sa["mlp.gate.weight"].dtype
            out_t["post_attention_layernorm.weight"] = torch.zeros(H, dtype=dtype)  # scale 1: folded
            out_t["mlp.gate.weight"] = moe.router.to(dtype)
            A_r, A_s = maps if maps is not None else (None, None)
            gus, dns = [], []
            for e in range(len(moe.experts)):
                gu, dn = moe.expert_weights(e)
                gus.append(gu.to(dtype))
                dns.append((A_r.T @ dn if A_r is not None else dn).to(dtype))
            out_t["mlp.experts.gate_up_proj"] = torch.stack(gus)
            out_t["mlp.experts.down_proj"] = torch.stack(dns)
            sh = moe.shared
            out_t["mlp.shared_expert.gate_proj.weight"] = sh["gate"].to(dtype)
            out_t["mlp.shared_expert.up_proj.weight"] = sh["up"].to(dtype)
            out_t["mlp.shared_expert.down_proj.weight"] = (
                A_s.T @ sh["down"] if A_s is not None else sh["down"]).to(dtype)
            out_t["mlp.shared_expert_gate.weight"] = sh["gate_vec"].to(dtype)
            pad_shared(out_t)
            write_layer(new_idx, out_t)
            new_idx += 1
            del sa, sb, mix_a, mix_b, moe_a, moe_b, mixers, moe, t0, t1, out_t, gus, dns

    # ---- held-out perplexity of the three streams on the calibration text ----
    norm_w = 1.0 + src.get(FINAL_NORM).float()
    head = src.get(LM_HEAD if LM_HEAD in src.weight_map else EMBED).float()
    nxt = ids[n_train:, 1:].reshape(-1)
    ppl = {}
    for name, stream in (("teacher", teacher), ("merged", merged), ("drop", drop)):
        h = (rms(stream[n_train:, :-1], eps) * norm_w).reshape(-1, H)
        nll = 0.0
        for lo in range(0, h.shape[0], 256):
            logits = h[lo:lo + 256] @ head.T
            nll += F.cross_entropy(logits, nxt[lo:lo + 256], reduction="sum").item()
        ppl[name] = round(math.exp(nll / nxt.numel()), 3)
    report["heldout_ppl"] = ppl
    print(f"\nheld-out calibration ppl: teacher {ppl['teacher']}  merged {ppl['merged']}  "
          f"drop (C2) {ppl['drop']}")
    print(f"note: {L}-token windows of the calibration text; not the same as llama.cpp's "
          f"wikitext ppl, compare the three against each other only.")

    if args.dry_run:
        print(json.dumps(report, indent=2))
        return report

    # ---- everything outside the language-model layers ----
    for name in src.weight_map:
        if LAYER_RE.match(name):
            continue
        t = src.get(name)
        if name.startswith("mtp.") and ".mlp.shared_expert." in name and t.dim() == 2:
            if name.endswith(("gate_proj.weight", "up_proj.weight")) and t.shape[0] != new_sf:
                t = torch.cat([t, t.new_zeros(new_sf - t.shape[0], t.shape[1])])
            elif name.endswith("down_proj.weight") and t.shape[1] != new_sf:
                t = torch.cat([t, t.new_zeros(t.shape[0], new_sf - t.shape[1])], dim=1)
        writer.add(name, t)
    n_shards, nbytes = writer.finish()

    tc = cfg_json["text_config"]
    tc["num_hidden_layers"] = new_idx
    interval = G - 1
    tc["full_attention_interval"] = interval
    tc["layer_types"] = [t for j, t in enumerate(types) if j % G != S + 1]
    tc["shared_expert_intermediate_size"] = new_sf
    tc["num_experts_per_tok"] = k_new
    json.dump(cfg_json, open(args.dst / "config.json", "w"), indent=2)
    for f in args.src.glob("*"):
        if f.suffix in (".json", ".txt", ".jinja") and not f.name.startswith("model") \
                and f.name != "config.json":
            shutil.copy(f, args.dst / f.name)
    json.dump(report, open(args.dst / "merge_report.json", "w"), indent=2)
    print(f"done: {new_idx} layers, {n_shards} shards, {nbytes / 1e9:.1f} GB -> {args.dst}")
    return report


if __name__ == "__main__":
    main()
