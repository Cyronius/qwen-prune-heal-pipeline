"""Shared pieces for the healing pipeline.

Everything that both the dataset build, the training run, and the merge step need
to agree on lives here, so they cannot drift apart.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

# Top-k teacher distribution width. 64 keeps the shards small enough to move around
# while still covering nearly all the probability mass for a 248k vocabulary.
TOP_K = 64

# LoRA targets. These are the nn.Linear modules on the LANGUAGE side only.
#
# The routed experts (gate_up_proj, down_proj) and the router (gate.weight) are raw
# nn.Parameter tensors, not Linear modules, so PEFT cannot reach them. Cut C leaves
# both untouched, so that is fine here. It is not fine for cut A -- see the plan.
#
# The prefix is optional because the two loader classes give different module trees:
#   Qwen3_5MoeForConditionalGeneration -> model.language_model.layers.N....
#   Qwen3_5MoeForCausalLM              -> model.layers.N....
#
# The vision tower has Linear layers too (qkv, proj, linear_fc1, linear_fc2), but they
# live under model.visual.blocks.N..., so `layers.\d+` never matches them. Verified,
# not assumed.
LORA_TARGET_REGEX = (
    r"^model\.(?:language_model\.)?layers\.\d+\."
    r"(?:"
    r"linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)"
    r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)"
    r"|mlp\.shared_expert_gate"
    r")$"
)

# transformers drops these on load (_keys_to_ignore_on_load_unexpected on the model
# class). They must be copied back by hand after merging or the healed model loses
# its multi-token-prediction head, and with it a chunk of decode speed.
MTP_PREFIX = "mtp."


def student_class(text_only=False):
    """Which class loads the checkpoint. This choice has consequences.

    AutoModelForCausalLM resolves to Qwen3_5MoeForCausalLM, which has no vision tower.
    It loads a checkpoint containing model.visual.* without complaint and then saves
    one without them. Combined with the mtp.* drop, loading and saving through that
    class quietly removes two whole subsystems.

    Default to the full ForConditionalGeneration class so the healed checkpoint keeps
    the same structure as the base we benchmarked against, and only the mtp.* head
    needs manual reattachment. Pass text_only=True to deliberately drop the vision
    tower and save its frozen weights from sitting in memory.
    """
    from transformers.models.qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )

    return Qwen3_5MoeForCausalLM if text_only else Qwen3_5MoeForConditionalGeneration


def load_config(path: str | Path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(path))


def tiny_config(reference: str | Path = None):
    """A few-hundred-thousand-parameter model with the same shape as the real one.

    Used by smoke.py to exercise the whole pipeline on CPU. Layer types keep the
    [linear, linear, full] pattern that cut C produced, so the code path under test
    is the same one the real run takes.
    """
    from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig

    cfg = copy.deepcopy(load_config(reference)) if reference else Qwen3_5MoeConfig()
    t = cfg.text_config
    t.hidden_size = 64
    t.intermediate_size = 128
    t.num_hidden_layers = 6
    t.full_attention_interval = 3
    t.layer_types = ["linear_attention", "linear_attention", "full_attention"] * 2
    t.num_experts = 8
    t.num_experts_per_tok = 2
    t.moe_intermediate_size = 32
    t.shared_expert_intermediate_size = 32
    t.vocab_size = 512
    t.head_dim = 32
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.linear_num_key_heads = 2
    t.linear_num_value_heads = 4
    t.linear_key_head_dim = 16
    t.linear_value_head_dim = 16
    v = cfg.vision_config
    v.hidden_size = 32
    v.depth = 2
    v.num_heads = 2
    v.out_hidden_size = 64
    if hasattr(v, "intermediate_size"):
        v.intermediate_size = 64
    return cfg


def build_lora(model, rank=32, alpha=64, dropout=0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_REGEX,
    )
    return get_peft_model(model, cfg)


def load_student(path, dtype=torch.bfloat16, four_bit=False, experts_impl="grouped_mm",
                 attn_impl=None, device_map="auto", text_only=False):
    """Load a pruned checkpoint as the student.

    experts_impl matters a lot. The default 'eager' path loops over all 256 experts in
    Python, one F.linear call each, which is far too slow to train with. 'grouped_mm'
    registers a real autograd function and batches the expert matmuls.
    """
    cls = student_class(text_only)

    kwargs = dict(dtype=dtype)
    if device_map:
        kwargs["device_map"] = device_map
    if experts_impl:
        kwargs["experts_implementation"] = experts_impl
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return cls.from_pretrained(str(path), **kwargs)


def freeze_vision(model):
    """The vision tower is dead weight for this project. Never train it."""
    n = 0
    for name, p in model.named_parameters():
        if ".visual." in name or name.startswith("visual."):
            p.requires_grad_(False)
            n += 1
    return n


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
