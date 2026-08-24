#!/bin/bash
set -uo pipefail
cd /c/code/model-shrink-ideas
LOG=heal-artifacts/finish.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
fail(){ log "FAILED: $1"; echo "$1" > heal-artifacts/FINISH-FAILED.flag; exit 1; }
BIN=bench/tools/llama

log "waiting for Q8 to finish"
while ! grep -q "main: quantize time" heal-artifacts/quant-q8.log 2>/dev/null; do sleep 60; done
log "Q8 done: $(du -sh qwen36-healed-C-Q8_0.gguf | cut -f1)"

log "imatrix over calibration.txt"
$BIN/llama-imatrix.exe -m qwen36-healed-C-Q8_0.gguf -f heal-artifacts/calibration.txt \
  -o heal-artifacts/healed-c.imatrix --chunks 120 -c 512 -ngl 99 >> $LOG 2>&1 || fail imatrix
log "imatrix done"

log "quantize bf16 -> Q4_K_M with imatrix"
$BIN/llama-quantize.exe --imatrix heal-artifacts/healed-c.imatrix \
  qwen36-healed-C-bf16.gguf qwen36-healed-C-Q4KM-imat.gguf Q4_K_M >> heal-artifacts/quant-q4.log 2>&1 || fail quantize
log "Q4 done: $(du -sh qwen36-healed-C-Q4KM-imat.gguf | cut -f1)"

log "registering healed-C in bench registry"
python - <<'PY' || exit 1
import json
r = json.load(open("bench/registry.json"))
r["models"]["healed-C"] = {
    "gguf": "C:/code/model-shrink-ideas/qwen36-healed-C-Q4KM-imat.gguf",
    "note": "cut-C prune + 550-step SFT heal (2026-08-20), imatrix Q4_K_M"}
json.dump(r, open("bench/registry.json", "w"), indent=2)
print("registered")
PY

log "benchmark: healed-C and base-q4km control, same session"
cd bench && python run_bench.py --models healed-C,base-q4km >> ../heal-artifacts/bench.log 2>&1 || fail bench
cd ..
log "ALL DONE"
touch heal-artifacts/FINISH-DONE.flag
