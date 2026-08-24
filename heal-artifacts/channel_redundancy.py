"""Does merging beat pruning for the expert width cut? Step 1: does redundancy exist?

Captures each expert's real post-gate intermediate activations -- SiLU(gate(x))*up(x),
512 channels wide -- for tokens the router actually sends it, using a forward-pre-hook
on the real Qwen3_5MoeExperts module. Nothing about the forward pass is touched or
reimplemented; the hook only observes (hidden_states, top_k_index, top_k_weights) and
recomputes the intermediate activations itself, read-only, from the module's own
weights.

Weight-space similarity was rejected on paper: two channels can point in different
directions and still compute nearly the same thing after the SiLU gate, or point the
same direction and diverge sharply once gated. Only activation correlation over real
data answers "would merging these break anything."

Runs on the 3-layer prefix (heal-artifacts/layer-prefix-3), which is numerically
identical in its experts to the full 40-layer base -- cut C's LoRA healing never
touched routed-expert weights (3-D Parameters, not nn.Linear).
"""
import sys
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:/code/model-shrink-ideas/heal-artifacts/layer-prefix-3"
CALIB_TXT = sys.argv[2] if len(sys.argv) > 2 else r"C:/code/model-shrink-ideas/heal-artifacts/calibration.txt"
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 24000
CHUNK_TOKENS = 1024  # forward pass window; keeps peak activation memory bounded

captured = defaultdict(list)  # (layer_name, expert_idx) -> list of [n_tok, 512] tensors


def make_hook(layer_name):
    def hook(module, args, kwargs):
        # forward(self, hidden_states, top_k_index, top_k_weights) -- accept either
        # calling convention.
        vals = list(args) + list(kwargs.values())
        hidden_states, top_k_index, top_k_weights = vals[0], vals[1], vals[2]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        with torch.no_grad():
            for expert_idx in torch.unique(top_k_index).tolist():
                if expert_idx >= module.num_experts:
                    continue
                token_idx, pos = torch.where(top_k_index == expert_idx)
                if token_idx.numel() == 0:
                    continue
                current_state = hidden_states[token_idx]
                gate, up = torch.nn.functional.linear(
                    current_state, module.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                h = module.act_fn(gate) * up  # [n_tok, 512] -- the thing we'd merge
                captured[(layer_name, expert_idx)].append(h.float().cpu())
        return None  # do not alter the real forward pass
    return hook


def main():
    print(f"loading {MODEL_DIR}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    n_hooked = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3_5MoeExperts):
            module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
            n_hooked += 1
    print(f"hooked {n_hooked} expert modules", flush=True)

    text = open(CALIB_TXT, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt")["input_ids"][0][:MAX_TOKENS]
    print(f"running {len(ids)} calibration tokens in chunks of {CHUNK_TOKENS}", flush=True)

    with torch.no_grad():
        for i in range(0, len(ids), CHUNK_TOKENS):
            chunk = ids[i:i + CHUNK_TOKENS].unsqueeze(0)
            model(input_ids=chunk)
            done = min(i + CHUNK_TOKENS, len(ids))
            print(f"  {done}/{len(ids)} tokens", flush=True)

    print(f"\ncaptured activity for {len(captured)} (layer, expert) pairs", flush=True)

    MIN_TOKENS = 24  # need enough samples for a correlation estimate to mean anything
    results = []
    for (layer_name, expert_idx), chunks in captured.items():
        h = torch.cat(chunks, dim=0)  # [n_tok, 512]
        if h.shape[0] < MIN_TOKENS:
            continue
        # drop channels that never fire -- SiLU(gate)*up == 0 everywhere is not
        # "perfectly correlated with everything," it's dead, and would swamp the
        # real signal with degenerate 1.0s.
        live = h.std(dim=0) > 1e-4
        h_live = h[:, live]
        if h_live.shape[1] < 2:
            continue
        corr = torch.corrcoef(h_live.T)  # [n_live, n_live]
        corr.fill_diagonal_(0)
        n = corr.shape[0]
        best = corr.abs().max(dim=1).values  # each channel's best partner, |r|
        results.append({
            "layer": layer_name, "expert": expert_idx, "n_tokens": h.shape[0],
            "n_live_channels": n, "n_dead_channels": 512 - n,
            "median_best_corr": best.median().item(),
            "frac_above_0.9": (best > 0.9).float().mean().item(),
            "frac_above_0.7": (best > 0.7).float().mean().item(),
            "max_corr": corr.abs().max().item(),
        })

    results.sort(key=lambda r: -r["n_tokens"])
    print(f"\n{len(results)} experts had >= {MIN_TOKENS} calibration tokens\n")
    print(f"{'layer':40s} {'exp':>4s} {'ntok':>5s} {'dead':>4s} {'med|r|':>7s} "
          f"{'>0.9':>6s} {'>0.7':>6s} {'max|r|':>6s}")
    for r in results[:20]:
        print(f"{r['layer']:40s} {r['expert']:4d} {r['n_tokens']:5d} "
              f"{r['n_dead_channels']:4d} {r['median_best_corr']:7.3f} "
              f"{r['frac_above_0.9']:6.1%} {r['frac_above_0.7']:6.1%} {r['max_corr']:6.3f}")

    if results:
        import statistics
        meds = [r["median_best_corr"] for r in results]
        p90 = [r["frac_above_0.9"] for r in results]
        print(f"\nacross all {len(results)} qualifying experts:")
        print(f"  median-best-|r|, averaged: {statistics.mean(meds):.3f}")
        print(f"  fraction of channels with a >0.9-correlated partner, averaged: {statistics.mean(p90):.1%}")
    print("\nDONE")


if __name__ == "__main__":
    main()
