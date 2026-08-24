#!/bin/bash
# Unattended full run: dataset -> train -> flag. Every phase logs; any failure
# writes FAILED.flag and stops. The pod's terminate-after wall is the backstop.
set -uo pipefail
W=/workspace
LOG=$W/run.log
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $LOG; }
fail(){ log "FAILED in phase $1"; echo "$1" > $W/FAILED.flag; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=1
cd $W/heal

log "=== phase env ==="
pip install -q --break-system-packages -U "transformers>=5.14.1" peft accelerate bitsandbytes safetensors datasets "huggingface_hub[hf_transfer]" 2>&1 | tail -1
pip install -q --break-system-packages flash-linear-attention tilelang 2>&1 | tail -1
pip install -q --break-system-packages causal-conv1d --no-build-isolation 2>&1 | tail -1 || log "WARN causal-conv1d"
python3 -c "import torch,causal_conv1d,fla,tilelang" || fail env

log "=== phase base ==="
hf download Qwen/Qwen3.6-35B-A3B --local-dir $W/base 2>&1 | tail -1 || fail base

log "=== phase prune ==="
python3 prune_qwen35.py $W/base $W/pruned-c --skip-expert-cut 2>&1 | tail -2 | tee -a $LOG || fail prune

log "=== phase prompts ==="
python3 make_prompts.py --out $W/prompts --total 18000 2>&1 | tail -12 | tee -a $LOG || fail prompts

log "=== phase generate (teacher completions, batched) ==="
python3 gen_teacher.py --teacher $W/base --mode generate \
  --input $W/prompts/prompts-generate.jsonl --out $W/gen-out \
  --max-new 768 --temperature 0.7 --batch 32 --log-every 5 2>&1 | tail -3 | tee -a $LOG || fail generate

log "=== phase forward-general (top-k over web text) ==="
python3 gen_teacher.py --teacher $W/base --mode forward \
  --input $W/prompts/prompts-general.jsonl --out $W/data-general \
  --max-len 4096 --log-every 200 2>&1 | tail -2 | tee -a $LOG || fail forward-general

log "=== phase forward-agentic (top-k over transcripts) ==="
python3 gen_teacher.py --teacher $W/base --mode forward \
  --input $W/gen-out/transcripts.jsonl --out $W/data-agentic \
  --max-len 4096 --log-every 200 2>&1 | tail -2 | tee -a $LOG || fail forward-agentic

log "=== phase train ==="
python3 train_heal.py --student $W/pruned-c --data $W/data-general $W/data-agentic \
  --out $W/heal-run --loss sft --four-bit --max-len 4096 \
  --batch-size 4 --grad-accum 4 --lr 1e-4 --warmup 30 \
  --steps 700 --eval-every 50 --save-every 100 2>&1 | tail -5 | tee -a $LOG || fail train

log "=== all phases complete ==="
touch $W/DONE.flag
