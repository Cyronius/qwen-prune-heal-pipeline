#!/bin/bash
# Gate session for the heal-C run. Attended; each phase logs wall-clock and cost.
# Usage: bash pod_gate.sh <phase>   (phases: env, base, prune, verify, gate)
set -euo pipefail
W=/workspace
LOG=$W/gate.log
RATE=3.59   # $/hr, H200 community
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }
phase_time(){ S=$SECONDS; }
phase_done(){ D=$((SECONDS-S)); log "phase $1 done in ${D}s (~\$$(python3 -c "print(f'{$D/3600*$RATE:.2f}')"))" ; }

case "${1:?phase required}" in
env)
  phase_time
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv | tee -a $LOG
  df -h $W | tee -a $LOG
  log "network probe:"; curl -so /dev/null -w '%{speed_download} B/s\n' https://speed.cloudflare.com/__down?bytes=100000000 | tee -a $LOG
  pip install -q -U "transformers>=5.14.1" peft accelerate bitsandbytes safetensors datasets "huggingface_hub[hf_transfer]" 2>&1 | tail -2
  pip install -q flash-linear-attention 2>&1 | tail -1 || log "WARN: fla install failed"
  pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -1 || log "WARN: causal-conv1d failed (GDN falls back to torch)"
  python3 -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| transformers',transformers.__version__)" | tee -a $LOG
  python3 -c "import torch;print('grouped_mm op present:',hasattr(torch,'_grouped_mm') or hasattr(torch.nn.functional,'grouped_mm'))" | tee -a $LOG
  phase_done env
  ;;
base)
  phase_time
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.6-35B-A3B --local-dir $W/base 2>&1 | tail -3
  du -sh $W/base | tee -a $LOG
  phase_done base
  ;;
prune)
  phase_time
  python3 $W/heal/prune_qwen35.py $W/base $W/pruned-c --skip-expert-cut 2>&1 | tail -5 | tee -a $LOG
  du -sh $W/pruned-c | tee -a $LOG
  phase_done prune
  ;;
verify)
  phase_time
  python3 - <<'PY' 2>&1 | tee -a $LOG
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
tok = AutoTokenizer.from_pretrained("/workspace/pruned-c")
m = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    "/workspace/pruned-c", dtype=torch.bfloat16, device_map="cuda",
    experts_implementation="grouped_mm")
msgs=[{"role":"user","content":"Name three primary colors."}]
ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
out=m.generate(ids, max_new_tokens=60, do_sample=False)
print("---VERIFY OUTPUT---")
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False))
PY
  phase_done verify
  ;;
gate)
  phase_time
  cd $W/heal
  python3 - <<'PY' 2>&1 | tee -a $LOG
# tiny forward-mode dataset from streamed fineweb, just enough for 20 timed steps
import json, itertools
from datasets import load_dataset
rows=[{"slice":"general","text":r["text"][:6000]}
      for r in itertools.islice(load_dataset("HuggingFaceFW/fineweb-edu","sample-10BT",split="train",streaming=True),60)]
open("/workspace/gate-prompts.jsonl","w").write("\n".join(json.dumps(r) for r in rows))
print(len(rows),"gate prompts")
PY
  python3 gen_teacher.py --teacher $W/pruned-c --mode forward \
      --input $W/gate-prompts.jsonl --out $W/gate-data --max-len 4096 --log-every 20 2>&1 | tail -3 | tee -a $LOG
  python3 train_heal.py --student $W/pruned-c --data $W/gate-data \
      --out $W/gate-run --loss sft --four-bit --max-len 4096 \
      --steps 20 --grad-accum 4 --eval-every 1000 --save-every 1000 --warmup 5 2>&1 | tee -a $LOG
  log "GATE: tok/s lines above; threshold is 800 tok/s"
  ;;
*) echo "unknown phase $1"; exit 1;;
esac
