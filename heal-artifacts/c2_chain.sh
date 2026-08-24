#!/bin/bash
set -uo pipefail
cd /c/code/model-shrink-ideas
LOG=heal-artifacts/c2.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
fail(){ log "FAILED: $1"; touch heal-artifacts/C2-FAILED.flag; exit 1; }

log "prune: 30 -> 20 layers, interval 3 -> 2 (drop GDN slot 1 of each 3-group)"
python prune_qwen35.py merged-heal-c pruned-healedC-c2 --skip-expert-cut --group 3 --drop-slot 1 >> $LOG 2>&1 || fail prune
log "prune done: $(du -sh pruned-healedC-c2 | cut -f1)"

log "convert (current llama.cpp, MTP head included)"
python /c/code/llama.cpp/convert_hf_to_gguf.py pruned-healedC-c2 \
  --outfile qwen36-healedC-c2-bf16.gguf --outtype bf16 >> heal-artifacts/c2-convert.log 2>&1 || fail convert

log "quantize plain Q4_K_M (stupid-run methodology; imatrix layer names would mismatch)"
bench/tools/llama-current/llama-quantize.exe qwen36-healedC-c2-bf16.gguf \
  qwen36-healedC-c2-Q4_K_M.gguf Q4_K_M >> heal-artifacts/c2-quant.log 2>&1 || fail quantize
python -c "import os; os.remove('qwen36-healedC-c2-bf16.gguf')"
log "quant done: $(du -sh qwen36-healedC-c2-Q4_K_M.gguf | cut -f1)"

python - <<'PY' || fail register
import json
r = json.load(open("bench/registry.json"))
r["models"]["healedC-c2"] = {"gguf": "C:/code/model-shrink-ideas/qwen36-healedC-c2-Q4_K_M.gguf",
  "note": "cut C squared: healed-C with 30->20 layers (interval 2), UNHEALED second cut, plain Q4_K_M"}
json.dump(r, open("bench/registry.json", "w"), indent=2)
PY
log "benchmarking"
cd bench && python run_bench.py --models healedC-c2 >> ../heal-artifacts/c2-bench.log 2>&1 || fail bench
cd ..
log "C2 ALL DONE"
touch heal-artifacts/C2-DONE.flag
