#!/bin/bash
set -uo pipefail
cd /c/code/model-shrink-ideas
LOG=heal-artifacts/w384.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
fail(){ log "FAILED: $1"; touch heal-artifacts/W384-FAILED.flag; exit 1; }

log "phase prune: healed-C experts 512->384"
python prune_qwen35.py merged-heal-c pruned-healedC-w384 --skip-layer-cut --new-ff 384 >> $LOG 2>&1 || fail prune
log "prune done: $(du -sh pruned-healedC-w384 | cut -f1)"

log "phase convert"
python /c/code/llama.cpp/convert_hf_to_gguf.py pruned-healedC-w384 \
  --outfile qwen36-healedC-w384-bf16.gguf --outtype bf16 >> heal-artifacts/w384-convert.log 2>&1 || fail convert
log "convert done"

log "phase quantize (plain Q4_K_M; imatrix is shape-bound to 512-wide experts)"
bench/tools/llama/llama-quantize.exe qwen36-healedC-w384-bf16.gguf \
  qwen36-healedC-w384-Q4_K_M.gguf Q4_K_M >> heal-artifacts/w384-quant.log 2>&1 || fail quantize
log "quant done: $(du -sh qwen36-healedC-w384-Q4_K_M.gguf | cut -f1)"
python -c "
import os; os.remove('qwen36-healedC-w384-bf16.gguf')"
log "bf16 intermediate deleted"

python - <<'PY' || fail register
import json
r = json.load(open("bench/registry.json"))
r["models"]["healedC-w384"] = {"gguf": "C:/code/model-shrink-ideas/qwen36-healedC-w384-Q4_K_M.gguf",
  "note": "healed-C with unhealed expert width cut 512->384, plain Q4_K_M"}
json.dump(r, open("bench/registry.json", "w"), indent=2)
PY
log "registered; benchmarking"
cd bench && python run_bench.py --models healedC-w384 >> ../heal-artifacts/w384-bench.log 2>&1 || fail bench
cd ..
log "W384 ALL DONE"
touch heal-artifacts/W384-DONE.flag
