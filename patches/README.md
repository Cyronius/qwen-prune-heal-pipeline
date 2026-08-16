# llama.cpp patches

## llamacpp-qwen35-tokenizer-hash.patch

Registers the Qwen3.6-35B-A3B tokenizer with `convert_hf_to_gguf.py`.

### Why it is needed

`convert_hf_to_gguf.py` identifies a model's pre-tokenizer by hashing it. If the hash
is unknown, the script refuses to convert and tells you to add it.

Qwen3.6-35B-A3B hashes to `1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f`.
Upstream does not know that hash. It uses the same pre-tokenizer as Qwen3.5-9B-Instruct,
so the patch maps it to the existing `qwen35` handler.

Without this patch, no GGUF in this project can be built. That includes the base
control, all three pruned models, and any healed model.

### How to apply

```
cd c:/code/llama.cpp
git apply c:/code/model-shrink-ideas/patches/llamacpp-qwen35-tokenizer-hash.patch
```

### Version it was taken from

| field | value |
|---|---|
| repo | c:/code/llama.cpp |
| upstream commit | fae3a28 (`ggml : remove ggml-ext.h (#21869)`) |
| file | convert_hf_to_gguf.py |
| exported | 2026-08-16 |

If upstream later adds this hash themselves, `git apply` will fail. Check whether the
hash is already present before assuming the patch is broken.

### The models this covers

The same tokenizer is shared by every checkpoint in this project, because they all
derive from the same base.

- `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B` (teacher)
- `pruned36-a-only/`
- `pruned36-c-only/`
- `pruned36-ac/`
