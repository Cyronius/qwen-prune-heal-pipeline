"""Merge the trained adapters back into a full bf16 checkpoint.

Two things happen here, and the second one is easy to forget.

1. The LoRA adapters are folded into the base weights and the result is saved as a
   normal checkpoint.

2. The mtp.* tensors are copied across from the source checkpoint.

Step 2 is necessary because Qwen3_5MoePreTrainedModel sets
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]. Transformers loads the model without
those 19 tensors and saves it without them too. Nothing warns you. The multi-token
prediction head just disappears, and with it the speculative-decoding speed win that
made the unsloth build the fastest baseline in Round-1.

Usage:
  python merge_heal.py --student pruned36-c-only --adapter runs/heal-c/adapter-final \
      --out merged/heal-c
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from common import MTP_PREFIX, student_class


def copy_mtp_tensors(src: Path, dst: Path):
    """Append the source checkpoint's mtp.* tensors to the merged output."""
    index_path = src / "model.safetensors.index.json"
    src_index = json.loads(index_path.read_text(encoding="utf-8"))
    mtp_names = [n for n in src_index["weight_map"] if n.startswith(MTP_PREFIX)]
    if not mtp_names:
        print("no mtp.* tensors in the source checkpoint; nothing to reattach")
        return 0

    tensors = {}
    for name in mtp_names:
        with safe_open(src / src_index["weight_map"][name], framework="pt") as f:
            # clone, or the mmap of the whole source shard stays pinned
            tensors[name] = f.get_tensor(name).clone()

    dst_index_path = dst / "model.safetensors.index.json"
    if dst_index_path.exists():
        dst_index = json.loads(dst_index_path.read_text(encoding="utf-8"))
    else:
        # save_pretrained wrote a single unsharded file. Build an index that points at
        # it, so the extra mtp shard can be added alongside. Do not rename the existing
        # file: transformers resolves weights through the index's weight_map, and a
        # rename only creates a chance to leave a dangling reference behind.
        weight_map = {}
        with safe_open(dst / "model.safetensors", framework="pt") as f:
            for k in f.keys():
                weight_map[k] = "model.safetensors"
        dst_index = {"metadata": {}, "weight_map": weight_map}

    shard_name = "model-mtp.safetensors"
    save_file(tensors, dst / shard_name, metadata={"format": "pt"})
    for name in mtp_names:
        dst_index["weight_map"][name] = shard_name
    total = sum((dst / f).stat().st_size for f in set(dst_index["weight_map"].values()))
    dst_index.setdefault("metadata", {})["total_size"] = total
    dst_index_path.write_text(json.dumps(dst_index, indent=2), encoding="utf-8")
    print(f"reattached {len(mtp_names)} mtp.* tensors as {shard_name}")
    return len(mtp_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="the pruned checkpoint that was trained")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts-impl", default="grouped_mm")
    ap.add_argument("--device-map", default="cpu")
    ap.add_argument("--skip-mtp", action="store_true")
    ap.add_argument("--text-only", action="store_true",
                    help="must match what train_heal.py used, or the adapter will not apply")
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoTokenizer

    src, dst = Path(args.student).resolve(), Path(args.out).resolve()
    # transformers treats an unreadable local directory as a Hub repo id and fails with
    # a confusing "Repo id must use alphanumeric chars" error. Check here instead.
    for path, what in ((src, "student"), (Path(args.adapter).resolve(), "adapter")):
        if not path.is_dir():
            raise SystemExit(f"{what} directory does not exist: {path}")
    if not (src / "model.safetensors.index.json").exists() and not (src / "model.safetensors").exists():
        raise SystemExit(f"student directory has no safetensors weights: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    print("loading base in bf16 (not 4-bit: merging into quantised weights loses the point)")
    kwargs = {"dtype": torch.bfloat16}
    if args.device_map and args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if args.experts_impl:
        kwargs["experts_implementation"] = args.experts_impl
    model = student_class(args.text_only).from_pretrained(str(src), **kwargs)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(str(dst), safe_serialization=True)

    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(dst))
    for extra in ("chat_template.jinja", "preprocessor_config.json", "generation_config.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)

    if not args.skip_mtp:
        copy_mtp_tensors(src, dst)

    print(f"merged checkpoint written to {dst}")
    print("next: convert to GGUF, then quantise with imatrix (not plain Q4_K_M)")


if __name__ == "__main__":
    main()
