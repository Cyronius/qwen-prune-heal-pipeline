#!/bin/bash
# C2-merge: same 30 -> 20 cut as c2_chain.sh, but each dropped GDN layer is merged into
# its neighbour (merge_layers_qwen35.py) instead of deleted. UNHEALED, plain Q4_K_M, so
# the bench numbers compare directly with c2-DEAD (tools .05, ppl 3741).
#
# Gate: the merge step reports held-out ppl for teacher / merged / plain-drop on the
# calibration text. If merged does not beat plain drop, stop before spending hours on
# convert + quantize + bench. FORCE=1 skips the gate.
#
# Disk: merged-c2m (~37 GB bf16) + bf16 GGUF (~37 GB, deleted after quantize) + Q4 (~12 GB).
set -uo pipefail
cd /c/code/model-shrink-ideas
LOG=heal-artifacts/c2m.log
log(){ echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
fail(){ log "FAILED: $1"; touch heal-artifacts/C2M-FAILED.flag; exit 1; }
rm -f heal-artifacts/C2M-*.flag

log "smoke test"
python merge_smoke.py >> $LOG 2>&1 || fail smoke

log "merge: 30 -> 20 layers, GDN pairs merged (default variant menu, per-group pick)"
python merge_layers_qwen35.py merged-heal-c merged-c2m \
  --calib heal-artifacts/calibration.txt >> $LOG 2>&1 || fail merge
log "merge done: $(du -sh merged-c2m | cut -f1)"

python - <<'PY' >> $LOG 2>&1
import json
r = json.load(open("merged-c2m/merge_report.json"))
p = r["heldout_ppl"]
print("held-out ppl:", p)
print("chosen per group:", [g["chosen"] for g in r["groups"]])
PY
if [ "${FORCE:-0}" != "1" ]; then
  python -c "import json,sys; p=json.load(open('merged-c2m/merge_report.json'))['heldout_ppl']; sys.exit(0 if p['merged'] < p['drop'] else 1)" \
    || { log "NO-GO: merged ppl is not below plain drop; stopping before convert (FORCE=1 to override)"; touch heal-artifacts/C2M-NOGO.flag; exit 0; }
fi

log "convert (current llama.cpp, MTP head included)"
python /c/code/llama.cpp/convert_hf_to_gguf.py merged-c2m \
  --outfile qwen36-c2m-bf16.gguf --outtype bf16 >> heal-artifacts/c2m-convert.log 2>&1 \
  || fail "convert (if it chokes on the 1024-wide shared expert, re-run the merge with --shared first)"

log "quantize plain Q4_K_M (same methodology as c2-DEAD)"
bench/tools/llama-current/llama-quantize.exe qwen36-c2m-bf16.gguf \
  qwen36-c2m-Q4_K_M.gguf Q4_K_M >> heal-artifacts/c2m-quant.log 2>&1 || fail quantize
python -c "import os; os.remove('qwen36-c2m-bf16.gguf')"
log "quant done: $(du -sh qwen36-c2m-Q4_K_M.gguf | cut -f1)"

python - <<'PY' || fail register
import json
r = json.load(open("bench/registry.json"))
rep = json.load(open("merged-c2m/merge_report.json"))
r["models"]["c2-merge"] = {"gguf": "C:/code/model-shrink-ideas/qwen36-c2m-Q4_K_M.gguf",
  "note": "30->20 like c2-DEAD, but GDN pairs MERGED (merge_layers_qwen35.py), UNHEALED, "
          f"plain Q4_K_M. calib ppl teacher/merged/drop: {rep['heldout_ppl']}"}
json.dump(r, open("bench/registry.json", "w"), indent=2)
PY
log "benchmarking"
cd bench && python run_bench.py --models c2-merge >> ../heal-artifacts/c2m-bench.log 2>&1 || fail bench
cd ..
log "C2M ALL DONE"
touch heal-artifacts/C2M-DONE.flag
