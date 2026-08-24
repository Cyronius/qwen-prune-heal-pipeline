"""Extract the first N language-model layers of a checkpoint, unmodified.

Not a cut -- a prefix. Layers keep their original weights and indices 0..N-1
untouched; everything after layer N-1 is simply not copied. Purpose: get real,
correctly-computed hidden states at shallow depth without materializing all 30
layers (52 GB) into RAM.

Safe specifically for merged-heal-c because cut C's LoRA healing never touched the
routed experts (they are 3-D nn.Parameter, LoRA can only reach nn.Linear) -- so the
expert weights in layers 0..N-1 here are byte-identical to the original 40-layer
base's. Any redundancy found in these experts is a real property of the base model,
not an artifact of pruning or healing.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
N_LAYERS = int(sys.argv[3])
DST.mkdir(parents=True, exist_ok=True)

index = json.load(open(SRC / "model.safetensors.index.json"))
weight_map = index["weight_map"]


def get_tensor(name):
    with safe_open(SRC / weight_map[name], framework="pt") as f:
        return f.get_tensor(name).clone()


keep_prefixes = tuple(f"model.language_model.layers.{i}." for i in range(N_LAYERS))
keep_names = [
    n for n in weight_map
    if n.startswith(keep_prefixes)
    or n in ("model.language_model.embed_tokens.weight",
             "model.language_model.norm.weight",
             "lm_head.weight")
]
print(f"keeping {len(keep_names)} tensors ({N_LAYERS} layers + embed/norm/lm_head)")

out = {}
for i, name in enumerate(sorted(keep_names)):
    print(f"  [{i}/{len(keep_names)}] {name}", flush=True)
    out[name] = get_tensor(name).contiguous()

save_file(out, DST / "model.safetensors", metadata={"format": "pt"})
total = (DST / "model.safetensors").stat().st_size
json.dump({"metadata": {"total_size": total},
          "weight_map": {k: "model.safetensors" for k in out}},
         open(DST / "model.safetensors.index.json", "w"), indent=2)

cfg = json.load(open(SRC / "config.json"))
cfg["text_config"]["num_hidden_layers"] = N_LAYERS
cfg["text_config"]["layer_types"] = cfg["text_config"]["layer_types"][:N_LAYERS]
json.dump(cfg, open(DST / "config.json", "w"), indent=2)

for f in SRC.glob("*"):
    if f.suffix in (".json", ".txt", ".jinja") and f.name not in ("config.json", "model.safetensors.index.json") and not f.name.startswith("model"):
        shutil.copy(f, DST / f.name)

print(f"done: {total/1e9:.2f} GB -> {DST}")
