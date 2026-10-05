"""CPU smoke test for merge_layers_qwen35.py, on a tiny random model with the real layout.

Checks the things that would silently ruin a real run:
  1. the script's layer-by-layer teacher forward matches the HF model's own forward
  2. for each variant, the checkpoint it writes, loaded back through HF, gives exactly
     the perplexity the script predicted for it (norm folding, router concat, shared
     concat, the folded output map, shard writing, config edits)
  3. "a:a:none" reproduces plain C2 drop exactly
  4. the MTP shared expert is zero-padded to the new width
  5. the result loads with no missing or unexpected keys

Takes about a minute. Run it after any change to the merge script.

  python merge_smoke.py [--work DIR]
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "heal"))
sys.path.insert(0, str(ROOT))

from common import tiny_config  # noqa: E402
import merge_layers_qwen35 as mlq  # noqa: E402

N_SEQ, SEQ_LEN, HELDOUT = 12, 48, 4


def build_source(work: Path):
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Qwen3_5MoeForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            # norms init to zero (scale 1), which would make norm folding trivially
            # exact; give them real values. Also make sure no expert is left uninitialised.
            if name.endswith("layernorm.weight") or name.endswith("model.language_model.norm.weight"):
                p.copy_(torch.randn_like(p) * 0.3)
            if "experts." in name:
                p.copy_(torch.randn_like(p) * 0.08)
            if name.endswith("mlp.gate.weight"):
                p.copy_(torch.randn_like(p) * 0.5)
    src = work / "src"
    model.save_pretrained(src, max_shard_size="200KB")
    # HF drops mtp.* on save; add a fake MTP shared expert so padding gets exercised
    t = cfg.text_config
    sf, hid = t.shared_expert_intermediate_size, t.hidden_size
    mtp = {"mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(sf, hid),
           "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(sf, hid),
           "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(hid, sf),
           "mtp.fc.weight": torch.randn(hid, 2 * hid)}
    save_file(mtp, src / "mtp.safetensors")
    idx = json.load(open(src / "model.safetensors.index.json"))
    idx["weight_map"].update({k: "mtp.safetensors" for k in mtp})
    json.dump(idx, open(src / "model.safetensors.index.json", "w"))
    return model, src


def hf_ppl(model, ids):
    with torch.no_grad():
        held = ids[N_SEQ - HELDOUT:]
        logits = model(input_ids=held).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), held[:, 1:].reshape(-1), reduction="mean")
    return math.exp(nll.item())


def load_back(dst):
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    m, info = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        dst, dtype=torch.float32, output_loading_info=True)
    # mtp.* shows up as unexpected on purpose: the class ignores it on load
    unexpected = [k for k in info["unexpected_keys"] if not k.startswith("mtp.")]
    assert not info["missing_keys"] and not info["mismatched_keys"] and not unexpected, \
        f"load problems: {info}"
    return m.eval()


def close(a, b, tol=2e-3):
    return abs(a - b) / max(abs(b), 1e-9) < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=None)
    work = ap.parse_args().work or Path(tempfile.mkdtemp(prefix="merge-smoke-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work}")

    teacher_model, src = build_source(work)
    torch.manual_seed(1)
    ids = torch.randint(0, teacher_model.config.text_config.vocab_size, (N_SEQ, SEQ_LEN))
    ref_teacher = hf_ppl(teacher_model, ids)
    common = ["--n-seq", str(N_SEQ), "--seq-len", str(SEQ_LEN), "--heldout", str(HELDOUT),
              "--batch", "3"]
    failures = []

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # 1 + 3: teacher path, and a:a:none == C2 drop
    dst = work / "out-drop"
    rep = mlq.main([str(src), str(dst), *common, "--force-variant", "a:a:none"], calib_ids=ids)
    ppl = rep["heldout_ppl"]
    check(close(ppl["teacher"], ref_teacher),
          f"teacher stream ppl {ppl['teacher']} == HF forward ppl {ref_teacher:.3f}")
    check(close(ppl["merged"], ppl["drop"]), f"a:a:none merged ppl {ppl['merged']} == drop ppl {ppl['drop']}")
    loaded = hf_ppl(load_back(dst), ids)
    check(close(loaded, ppl["merged"]), f"a:a:none checkpoint ppl {loaded:.3f} == predicted {ppl['merged']}")

    # 2: every kind of merge round-trips through the written checkpoint
    for v in ["a:union:full", "b:union:diag", "avg:union:none", "avg:avg:none", "a:a:full"]:
        dst = work / ("out-" + v.replace(":", "_"))
        rep = mlq.main([str(src), str(dst), *common, "--force-variant", v, "--top-k", "3"],
                       calib_ids=ids)
        loaded = hf_ppl(load_back(dst), ids)
        check(close(loaded, rep["heldout_ppl"]["merged"]),
              f"{v:15s} checkpoint ppl {loaded:.3f} == predicted {rep['heldout_ppl']['merged']}")

    # the default menu runs end to end and picks per group
    dst = work / "out-auto"
    rep = mlq.main([str(src), str(dst), *common], calib_ids=ids)
    loaded = hf_ppl(load_back(dst), ids)
    check(close(loaded, rep["heldout_ppl"]["merged"]),
          f"auto-picked ({[g['chosen'] for g in rep['groups']]}) checkpoint ppl {loaded:.3f} "
          f"== predicted {rep['heldout_ppl']['merged']}")

    # 4 + config
    cfg = json.load(open(dst / "config.json"))["text_config"]
    sf = teacher_model.config.text_config.shared_expert_intermediate_size
    check(cfg["num_hidden_layers"] == 4 and cfg["full_attention_interval"] == 2
          and cfg["layer_types"] == ["linear_attention", "full_attention"] * 2,
          f"config: {cfg['num_hidden_layers']} layers, interval {cfg['full_attention_interval']}")
    check(cfg["shared_expert_intermediate_size"] == 2 * sf, "config: shared expert width doubled")
    idx = json.load(open(dst / "model.safetensors.index.json"))["weight_map"]
    name = "mtp.layers.0.mlp.shared_expert.down_proj.weight"
    t = load_file(dst / idx[name])[name]
    src_t = load_file(src / "mtp.safetensors")[name]
    check(t.shape[1] == 2 * sf and torch.equal(t[:, :sf], src_t) and not t[:, sf:].any(),
          f"MTP shared expert padded {tuple(src_t.shape)} -> {tuple(t.shape)}, zeros in the pad")
    check((dst / "merge_report.json").exists(), "merge_report.json written")

    print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
