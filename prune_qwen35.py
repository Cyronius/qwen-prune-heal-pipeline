"""Aggressive no-heal prune of Qwen3.5-35B-A3B (cuts A + C from aggressive-multiple-layer-prune.md).

Cut A: halve expert intermediate 512 -> 256 (routed + shared + MTP experts).
       Channel selection per expert: keep top-256 by ||gate_c|| * ||up_c|| * ||down_c|| (fp32 norms).
Cut C: drop 1 of every 3 GDN layers (old layer index i where i % 4 == 1),
       40 -> 30 layers, full_attention_interval 4 -> 3. Attention layers untouched.

Streams shard-by-shard via safetensors lazy loading; peak RAM ~6GB.
Vision tower copied unchanged. No healing. That's the point.
"""
import faulthandler
import gc
import json
import re
import sys
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

faulthandler.enable()

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
DO_CUT_A = "--skip-expert-cut" not in sys.argv  # C-only control run when skipped
DO_CUT_C = "--skip-layer-cut" not in sys.argv   # A-only run when skipped
DST.mkdir(parents=True, exist_ok=True)

# source layer count read from config, so the script can re-cut an already-C-cut model
_src_cfg = json.load(open(SRC / "config.json"))
OLD_LAYERS = _src_cfg["text_config"]["num_hidden_layers"]
# Layer groups: the source pattern repeats every GROUP layers with the attention layer
# last ((i+1) % interval != 0). Dropping slot S from each group must hit a GDN, i.e.
# S in [0, GROUP-2]. 40-layer base: GROUP 4. A 30-layer interval-3 model: GROUP 3.
GROUP = int(sys.argv[sys.argv.index("--group") + 1]) if "--group" in sys.argv else 4
DROP_SLOT = int(sys.argv[sys.argv.index("--drop-slot") + 1]) if "--drop-slot" in sys.argv else 1
assert DROP_SLOT < GROUP - 1, "drop slot must be a GDN position, not the attention layer"
NEW_FF = int(sys.argv[sys.argv.index("--new-ff") + 1]) if "--new-ff" in sys.argv else 256
OLD_FF = _src_cfg["text_config"]["moe_intermediate_size"]
SHARD_BYTES = 1 * 1024**3  # small shards: Windows commit headroom is tight

keep_layers = [i for i in range(OLD_LAYERS) if not DO_CUT_C or i % GROUP != DROP_SLOT]
old2new = {old: new for new, old in enumerate(keep_layers)}
print(f"keeping {len(keep_layers)}/{OLD_LAYERS} layers; dropping {[i for i in range(OLD_LAYERS) if i % GROUP == DROP_SLOT]}")

index = json.load(open(SRC / "model.safetensors.index.json"))
weight_map = index["weight_map"]

def get_tensor(name):
    with safe_open(SRC / weight_map[name], framework="pt") as f:
        return f.get_tensor(name)

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

def keep_indices(gate, up, down):
    """gate/up: [ff, embd], down: [embd, ff]. Returns sorted top-NEW_FF channel indices."""
    score = (gate.float().norm(dim=1) * up.float().norm(dim=1) * down.float().norm(dim=0))
    return torch.topk(score, NEW_FF).indices.sort().values

# ---- pass 1: build the output tensor plan (name -> producer fn), grouping fused experts ----
out_tensors = {}  # new_name -> callable returning tensor

def plan_copy(new_name, src_name):
    out_tensors[new_name] = lambda n=src_name: get_tensor(n)

def plan_fused_experts(new_prefix, src_prefix):
    if not DO_CUT_A:
        plan_copy(new_prefix + "gate_up_proj", src_prefix + "gate_up_proj")
        plan_copy(new_prefix + "down_proj", src_prefix + "down_proj")
        return
    def make_gate_up(p=src_prefix):
        gu = get_tensor(p + "gate_up_proj")          # [n_exp, 2*ff, embd]
        dn = get_tensor(p + "down_proj")             # [n_exp, embd, ff]
        n_exp = gu.shape[0]
        out = torch.empty((n_exp, 2 * NEW_FF, gu.shape[2]), dtype=gu.dtype)
        for e in range(n_exp):
            idx = keep_indices(gu[e, :OLD_FF], gu[e, OLD_FF:], dn[e])
            out[e, :NEW_FF] = gu[e, :OLD_FF][idx]
            out[e, NEW_FF:] = gu[e, OLD_FF:][idx]
        return out
    def make_down(p=src_prefix):
        gu = get_tensor(p + "gate_up_proj")
        dn = get_tensor(p + "down_proj")
        n_exp = gu.shape[0]
        out = torch.empty((n_exp, dn.shape[1], NEW_FF), dtype=dn.dtype)
        for e in range(n_exp):
            idx = keep_indices(gu[e, :OLD_FF], gu[e, OLD_FF:], dn[e])
            out[e] = dn[e][:, idx]
        return out
    out_tensors[new_prefix + "gate_up_proj"] = make_gate_up
    out_tensors[new_prefix + "down_proj"] = make_down

def plan_split_expert(new_prefix, src_prefix):
    """Unfused expert (shared expert / MTP experts): gate_proj/up_proj [ff,embd], down_proj [embd,ff]."""
    if not DO_CUT_A:
        for s in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            plan_copy(new_prefix + s, src_prefix + s)
        return
    def idx_of(p=src_prefix):
        return keep_indices(get_tensor(p + "gate_proj.weight"),
                            get_tensor(p + "up_proj.weight"),
                            get_tensor(p + "down_proj.weight"))
    out_tensors[new_prefix + "gate_proj.weight"] = lambda p=src_prefix: get_tensor(p + "gate_proj.weight")[idx_of(p)]
    out_tensors[new_prefix + "up_proj.weight"]   = lambda p=src_prefix: get_tensor(p + "up_proj.weight")[idx_of(p)]
    out_tensors[new_prefix + "down_proj.weight"] = lambda p=src_prefix: get_tensor(p + "down_proj.weight")[:, idx_of(p)]

handled = set()
for name in weight_map:
    if name in handled:
        continue
    m = LAYER_RE.match(name)
    if m:
        old = int(m.group(1))
        if old not in old2new:
            continue  # dropped layer
        new_layer_prefix = f"model.language_model.layers.{old2new[old]}."
        rename = lambda s, p=new_layer_prefix: LAYER_RE.sub(p, s)
    else:
        rename = lambda s: s  # mtp.*, visual.*, embed, lm_head, norms

    # fused experts (main layers; mtp too on Qwen3.6, where mtp experts are fused)
    if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
        prefix = name.rsplit("gate_up_proj", 1)[0] if name.endswith("gate_up_proj") else name.rsplit("down_proj", 1)[0]
        if rename(prefix) + "gate_up_proj" not in out_tensors:
            plan_fused_experts(rename(prefix), prefix)
        handled.add(prefix + "gate_up_proj"); handled.add(prefix + "down_proj")
    # unfused single experts: shared experts everywhere; per-expert mtp tensors on Qwen3.5
    elif (".mlp.shared_expert." in name or ".mlp.experts." in name) and name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")):
        prefix = name.rsplit(name.split(".")[-2] + ".weight", 1)[0]
        if rename(prefix) + "gate_proj.weight" not in out_tensors:
            plan_split_expert(rename(prefix), prefix)
        for s in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            handled.add(prefix + s)
    else:
        plan_copy(rename(name), name)

print(f"planned {len(out_tensors)} output tensors")

# ---- pass 2: materialize and shard ----
RESUME_INDEX = DST / "resume_index.json"
new_weight_map = {}
shard_buf, shard_size, shard_id, shard_files = {}, 0, 0, []

def flush():
    global shard_buf, shard_size, shard_id
    if not shard_buf:
        return
    fname = f"model-{shard_id:05d}.safetensors"
    save_file(shard_buf, DST / fname, metadata={"format": "pt"})
    for k in shard_buf:
        new_weight_map[k] = fname
    shard_files.append(fname)
    json.dump({"weight_map": new_weight_map}, open(RESUME_INDEX, "w"))
    print(f"  wrote {fname} ({shard_size/1e9:.2f} GB, {len(shard_buf)} tensors)", flush=True)
    shard_buf, shard_size, shard_id = {}, 0, shard_id + 1

done = set()
if RESUME_INDEX.exists():
    prev = json.load(open(RESUME_INDEX))
    done = set(prev["weight_map"])
    new_weight_map.update(prev["weight_map"])
    shard_files.extend(sorted(set(prev["weight_map"].values())))
    shard_id = len(shard_files)
    print(f"resuming: {len(done)} tensors already written in {shard_id} shards")

for i, (name, fn) in enumerate(sorted(out_tensors.items())):
    if name in done:
        continue
    print(f"  [{i}/{len(out_tensors)}] {name}", flush=True)
    # .clone() forces a real copy — otherwise mmap-backed tensors pin their
    # whole source shard's copy-on-write mapping until flush (commit blowup)
    t = fn().contiguous().clone()
    del fn
    shard_buf[name] = t
    if i % 10 == 0:
        gc.collect()
    shard_size += t.numel() * t.element_size()
    if shard_size >= SHARD_BYTES:
        flush()
flush()

# rename shards to the count-aware convention and build index
total = len(shard_files)
final_map = {}
for j, fname in enumerate(shard_files):
    final = f"model-{j+1:05d}-of-{total:05d}.safetensors"
    shutil.move(DST / fname, DST / final)
    for k, v in new_weight_map.items():
        if v == fname:
            final_map[k] = final
RESUME_INDEX.unlink(missing_ok=True)
total_bytes = sum((DST / f).stat().st_size for f in set(final_map.values()))
json.dump({"metadata": {"total_size": total_bytes}, "weight_map": final_map},
          open(DST / "model.safetensors.index.json", "w"), indent=2)

# ---- config + tokenizer ----
cfg = json.load(open(SRC / "config.json"))
tc = cfg["text_config"]
tc["num_hidden_layers"] = len(keep_layers)
if DO_CUT_C:
    new_interval = GROUP - 1
    tc["full_attention_interval"] = new_interval
    tc["layer_types"] = ((["linear_attention"] * (new_interval - 1) + ["full_attention"])
                         * (len(keep_layers) // new_interval))
if DO_CUT_A:
    tc["moe_intermediate_size"] = NEW_FF
    tc["shared_expert_intermediate_size"] = NEW_FF
json.dump(cfg, open(DST / "config.json", "w"), indent=2)

for f in SRC.glob("*"):
    if f.suffix in (".json", ".txt", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json") and not f.name.startswith("model"):
        shutil.copy(f, DST / f.name)

print(f"done: {total} shards, {total_bytes/1e9:.1f} GB -> {DST}")
