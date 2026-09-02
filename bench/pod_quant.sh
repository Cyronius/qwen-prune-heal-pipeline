#!/bin/bash
# Build extra quants of Qwen3.8-Flash-Next-131B-A6B (keep-1 n-gram surgery) on a
# CPU pod — no GPU needed, the surgery is a byte-copy. Phases are flag-gated so a
# rerun resumes. Any failure writes FAILED.flag.
# Layout: /workspace/{repo,src,builds,flags}
#
# Usage:
#   bash pod_quant.sh                  # builds Q4_K_M (the default)
#   bash pod_quant.sh Q4_K_M UD-Q4_K_XL
#   UPLOAD_REPO=Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF bash pod_quant.sh
#   SMOKE=1 bash pod_quant.sh          # + load-and-generate check (needs ~85 GB RAM)
#
# Disk: source Q4_K_M is ~100 GB, output ~75-80 GB — provision >= 250 GB per quant
# (or export PRUNE_SRC=1 to delete each source after its build, ~200 GB then).
# Upload needs `hf auth login` (or HF_TOKEN exported) on the pod first.
set -uo pipefail
W=${W:-/workspace}
LOG=$W/run.log
QUANTS=("${@:-Q4_K_M}")
HF_SRC=unsloth/Qwen3.8-Flash-Next-GGUF
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $LOG; }
fail(){ log "FAILED in phase $1"; echo "$1" > $W/FAILED.flag; exit 1; }
done_flag(){ [ -f $W/flags/$1.done ]; }
mark(){ mkdir -p $W/flags; touch $W/flags/$1.done; }
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $W/builds
cd $W

if ! done_flag env; then
  log "=== phase env ==="
  pip install -q --break-system-packages -U gguf numpy "huggingface_hub[cli,hf_transfer]" 2>&1 | tail -1
  python3 -c "import gguf, numpy" || fail env
  mark env
fi

if ! done_flag repo; then
  log "=== phase repo ==="
  [ -d $W/repo ] || git clone --depth 1 https://github.com/Cyronius/qwen-prune-heal-pipeline $W/repo 2>&1 | tail -1 || fail repo
  [ -f $W/repo/surgery_qwen38.py ] || fail repo
  mark repo
fi

for q in "${QUANTS[@]}"; do
  name="qwen38-keep1-$(echo "$q" | sed 's/^UD-//; s/_//g')"   # Q4_K_M -> qwen38-keep1-Q4KM

  if ! done_flag "download-$q"; then
    log "=== phase download $q ==="
    hf download $HF_SRC --include "$q/*" --local-dir $W/src 2>&1 | tail -1 || fail "download-$q"
    mark "download-$q"
  fi

  # first shard of a split GGUF, or the lone .gguf
  SRC=$(ls $W/src/$q/*-00001-of-*.gguf 2>/dev/null || ls $W/src/$q/*.gguf 2>/dev/null | head -1)
  [ -n "$SRC" ] || fail "locate-src-$q"

  if ! done_flag "build-$name"; then
    log "=== phase build $name ==="
    python3 $W/repo/surgery_qwen38.py "$SRC" /dev/null --ple-keep 1 --dry-run 2>&1 | tee -a $LOG || fail "dryrun-$name"
    need_kb=$(python3 $W/repo/surgery_qwen38.py "$SRC" /dev/null --ple-keep 1 --dry-run 2>/dev/null \
              | sed -n 's/.*(\([0-9.]*\) GB decimal.*/\1/p' | awk '{printf "%d", $1*1024*1024}')
    avail_kb=$(df --output=avail -k $W | tail -1 | tr -d ' ')
    [ "$avail_kb" -gt "$need_kb" ] || fail "disk-$name"
    python3 $W/repo/surgery_qwen38.py "$SRC" "$W/builds/$name.gguf" --ple-keep 1 2>&1 | tail -3 | tee -a $LOG || fail "build-$name"
    mark "build-$name"
  fi

  if [ -n "${SMOKE:-}" ] && ! done_flag "smoke-$name"; then
    log "=== phase smoke $name (CPU llama.cpp, slow but decisive) ==="
    if [ ! -x $W/llama.cpp/build/bin/llama-cli ]; then
      apt-get install -y -qq cmake ccache > /dev/null 2>&1 || true
      [ -d $W/llama.cpp ] || git clone --depth 1 --branch b10673 https://github.com/ggml-org/llama.cpp $W/llama.cpp 2>&1 | tail -1
      cmake -S $W/llama.cpp -B $W/llama.cpp/build -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 || fail smoke-cmake
      cmake --build $W/llama.cpp/build -j "$(nproc)" --target llama-cli 2>&1 | tail -1 || fail smoke-build
    fi
    $W/llama.cpp/build/bin/llama-cli -m "$W/builds/$name.gguf" -no-cnv --temp 0 -n 16 \
      -p "The capital of France is" 2>&1 | tail -5 | tee -a $LOG || fail "smoke-$name"
    mark "smoke-$name"
  fi

  if [ -n "${UPLOAD_REPO:-}" ] && ! done_flag "upload-$name"; then
    log "=== phase upload $name -> $UPLOAD_REPO ==="
    hf upload "$UPLOAD_REPO" "$W/builds/$name.gguf" "$name.gguf" 2>&1 | tail -1 || fail "upload-$name"
    mark "upload-$name"
  fi

  [ -n "${PRUNE_SRC:-}" ] && rm -rf "$W/src/$q" && log "pruned source $q"
done

log "=== all phases complete ==="
touch $W/DONE.flag
