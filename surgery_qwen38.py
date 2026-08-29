#!/usr/bin/env python
"""GGUF->GGUF surgery for Qwen3.8-Flash-Next (arch qwen4exp). No dequantization:
quantized bytes are copied verbatim, so unsloth's imatrix calibration is inherited.

Levers (composable in one pass):
  --ple-keep K   keep the first K of the 8 hash heads per n-gram type (2 types),
                 drop the rest: dropped heads get vocab_size=1 pointing at one
                 shared zero row (IQ4_NL all-zero bytes decode to zeros), which
                 needs NO llama.cpp patch. K=8 no-op, K=4 halves the table,
                 K=2 quarters it, K=0 ablates it entirely.
  --cut-layers   48 -> 36: drop one GDN layer per [G,G,G,F] group of 4
                 (full_attention_interval 4 -> 3, the proven 3.6 cut-C shape).
                 Group 0 drops slot 2 because layer 1 is the PLE injection
                 layer and must survive; groups 1..11 drop the middle GDN
                 (slot 1) to match the 3.6 recipe.

Input is the first shard of a split GGUF (siblings are found automatically) or a
single-file GGUF. Output is always a single file.

Corpus-remap vocab shrink (smaller prime per head, frequency-weighted row merge)
is a possible future mode; not implemented here.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

ALIGN_DEFAULT = 32
PLE_TENSOR = "per_layer_token_embd.weight"
K_HEAD_OFFSETS = "qwen4exp.ple.head_offsets"
K_HEAD_VOCABS = "qwen4exp.ple.head_vocab_sizes"
K_BLOCK_COUNT = "qwen4exp.block_count"
K_ATTN_INTERVAL = "qwen4exp.full_attention_interval"
K_PLE_LAYERS = "qwen4exp.ple.layers"
CHUNK = 256 * 1024 * 1024  # bytes per write


def open_shards(first: Path) -> list[GGUFReader]:
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [GGUFReader(first)]
    stem, _, total = m.groups()
    paths = [first.parent / f"{stem}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]
    for p in paths:
        if not p.exists():
            sys.exit(f"missing shard: {p}")
    return [GGUFReader(p) for p in paths]


def field_value(field):
    sub = field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None
    return field.contents(), field.types[0], sub


def raw_bytes(tensor) -> np.ndarray:
    """Flat uint8 view of a ReaderTensor's data, whatever dtype gguf-py mapped it to."""
    return tensor.data.reshape(-1).view(np.uint8)


def plan_ple(readers, keep_per_gram: int):
    """Return (new_offsets, new_vocabs, segments, new_rows, row_bytes).
    segments = list of (src_row_start, n_rows) byte-copy ranges, or ('zero', n_rows)."""
    fields = readers[0].fields
    offsets = [int(v) for v in fields[K_HEAD_OFFSETS].contents()]
    vocabs = [int(v) for v in fields[K_HEAD_VOCABS].contents()]
    per_gram = int(fields["qwen4exp.ple.heads_per_ngram"].contents())
    n_heads = len(offsets)
    n_grams = n_heads // per_gram
    keep = {g * per_gram + j for g in range(n_grams) for j in range(keep_per_gram)}

    new_offsets, new_vocabs, segments = [], [], []
    cursor = 0
    for h in range(n_heads):
        if h in keep:
            new_offsets.append(cursor)
            new_vocabs.append(vocabs[h])
            segments.append((offsets[h], vocabs[h]))
            cursor += vocabs[h]
        else:
            new_offsets.append(-1)  # patched to zero-row index below
            new_vocabs.append(1)
    zero_row = cursor
    segments.append(("zero", 1))
    cursor += 1
    new_offsets = [zero_row if o < 0 else o for o in new_offsets]
    return new_offsets, new_vocabs, segments, cursor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="first shard (or single file) of source GGUF")
    ap.add_argument("output", type=Path, help="output single-file GGUF")
    ap.add_argument("--ple-keep", type=int, default=8, choices=range(0, 9),
                    help="hash heads to keep per n-gram type (8 = untouched)")
    ap.add_argument("--cut-layers", action="store_true", help="48->36 GDN layer cut")
    ap.add_argument("--dry-run", action="store_true", help="report sizes, write nothing")
    args = ap.parse_args()

    readers = open_shards(args.input)
    r0 = readers[0]
    arch = r0.fields["general.architecture"].contents()
    assert arch == "qwen4exp", f"unexpected arch {arch}"

    n_blocks = int(r0.fields[K_BLOCK_COUNT].contents())
    interval = int(r0.fields[K_ATTN_INTERVAL].contents())
    ple_layers = [int(v) for v in r0.fields[K_PLE_LAYERS].contents()]

    # ---- layer cut plan ----
    layer_map = {i: i for i in range(n_blocks)}  # old -> new
    dropped_layers: set[int] = set()
    if args.cut_layers:
        assert n_blocks == 48 and interval == 4, "cut recipe expects the stock 48/4 layout"
        for g in range(n_blocks // interval):
            victim = g * interval + (2 if g == 0 else 1)
            assert victim not in ple_layers, "refusing to drop the PLE layer"
            dropped_layers.add(victim)
        kept = [i for i in range(n_blocks) if i not in dropped_layers]
        layer_map = {old: new for new, old in enumerate(kept)}
        new_ple_layers = [layer_map[l] for l in ple_layers]
    else:
        new_ple_layers = ple_layers

    # ---- PLE plan ----
    do_ple = args.ple_keep < 8
    if do_ple:
        new_offsets, new_vocabs, ple_segments, new_rows = plan_ple(readers, args.ple_keep)

    # ---- gather tensors across shards, apply drop/rename ----
    blk_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    out_tensors = []  # (out_name, reader_tensor)
    seen_arrays_48 = []
    for rd in readers:
        for t in rd.tensors:
            m = blk_re.match(t.name)
            if m:
                old = int(m.group(1))
                if old in dropped_layers:
                    continue
                out_tensors.append((f"blk.{layer_map[old]}.{m.group(2)}", t))
            else:
                out_tensors.append((t.name, t))

    # ---- size report ----
    row_bytes = None
    total = 0
    for name, t in out_tensors:
        nb = int(t.n_bytes)
        if name == PLE_TENSOR:
            qtype = GGMLQuantizationType(t.tensor_type)
            blk_elems, blk_bytes = GGML_QUANT_SIZES[qtype]
            head_dim = int(t.shape[0])
            assert head_dim % blk_elems == 0, "PLE rows not block-aligned; cannot byte-slice"
            row_bytes = head_dim // blk_elems * blk_bytes
            if do_ple:
                nb = new_rows * row_bytes
        total += nb
    print(f"planned tensor payload: {total / 2**30:.1f} GiB "
          f"({total / 1e9:.1f} GB decimal, + ~10 MB metadata)")
    if args.dry_run:
        return

    # ---- write ----
    w = GGUFWriter(args.output, arch)
    skip_keys = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for field in r0.fields.values():
        if field.name in skip_keys or field.name.startswith("split."):
            continue
        val, vtype, sub = field_value(field)
        if vtype == GGUFValueType.ARRAY and hasattr(val, "__len__") and len(val) == n_blocks \
                and args.cut_layers and field.name not in (K_HEAD_OFFSETS, K_HEAD_VOCABS):
            # per-layer metadata: keep the surviving layers' entries, in order
            val = [val[old] for old in sorted(layer_map)]
            seen_arrays_48.append(field.name)
        if field.name == K_BLOCK_COUNT and args.cut_layers:
            val = 48 - len(dropped_layers)
        elif field.name == K_ATTN_INTERVAL and args.cut_layers:
            val = interval - 1
        elif field.name == K_PLE_LAYERS:
            val = new_ple_layers
        elif field.name == K_HEAD_OFFSETS and do_ple:
            val = new_offsets
        elif field.name == K_HEAD_VOCABS and do_ple:
            val = new_vocabs
        w.add_key_value(field.name, val, vtype, sub_type=sub)
    if seen_arrays_48:
        print(f"per-layer metadata arrays subset {n_blocks}->{48 - len(dropped_layers)}: {seen_arrays_48}")

    for name, t in out_tensors:
        qtype = GGMLQuantizationType(t.tensor_type)
        shape = [int(d) for d in t.shape]
        nb = int(t.n_bytes)
        if name == PLE_TENSOR and do_ple:
            shape = [shape[0], new_rows]
            nb = new_rows * row_bytes
        # add_tensor_info with raw_dtype wants numpy-order shape with the last
        # axis in BYTES (it converts back via quant_shape_from_byte_shape)
        blk_e, blk_b = GGML_QUANT_SIZES[qtype]
        assert shape[0] % blk_e == 0, (name, shape, qtype)
        np_shape = list(reversed(shape[1:])) + [shape[0] // blk_e * blk_b]
        w.add_tensor_info(name, np_shape, np.dtype(np.uint8), nb, raw_dtype=qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    fout = w.fout[0]
    align = ALIGN_DEFAULT

    def pad():
        pos = fout.tell()
        need = (align - pos % align) % align
        if need:
            fout.write(b"\x00" * need)

    def write_chunked(view: np.ndarray):
        for i in range(0, view.size, CHUNK):
            fout.write(view[i:i + CHUNK].tobytes())

    done = 0
    for name, t in out_tensors:
        pad()
        if name == PLE_TENSOR and do_ple:
            src = raw_bytes(t)
            for seg in ple_segments:
                if seg[0] == "zero":
                    fout.write(b"\x00" * (seg[1] * row_bytes))
                else:
                    start, n = seg
                    write_chunked(src[start * row_bytes:(start + n) * row_bytes])
        else:
            write_chunked(raw_bytes(t))
        done += 1
        if done % 100 == 0 or name == PLE_TENSOR:
            print(f"  [{done}/{len(out_tensors)}] {name}", flush=True)
    fout.close()

    # ---- self-check ----
    rv = GGUFReader(args.output)
    assert len(rv.tensors) == len(out_tensors)
    if do_ple:
        got = [int(v) for v in rv.fields[K_HEAD_VOCABS].contents()]
        assert got == new_vocabs, got
        ple = next(t for t in rv.tensors if t.name == PLE_TENSOR)
        assert int(ple.shape[1]) == new_rows, ple.shape
    if args.cut_layers:
        assert int(rv.fields[K_BLOCK_COUNT].contents()) == 36
        assert int(rv.fields[K_ATTN_INTERVAL].contents()) == 3
    # spot-check one copied tensor byte-for-byte
    name0, t0 = next((n, t) for n, t in out_tensors if n != PLE_TENSOR and t.n_bytes < 10 * 2**20)
    v0 = next(t for t in rv.tensors if t.name == name0)
    assert bytes(raw_bytes(v0)[:4096]) == bytes(raw_bytes(t0)[:4096]), f"byte mismatch in {name0}"
    print(f"OK: {args.output} ({args.output.stat().st_size / 1e9:.1f} GB), self-check passed")


if __name__ == "__main__":
    main()
