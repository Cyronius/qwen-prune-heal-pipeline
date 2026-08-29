#!/bin/bash
# Qwen3.8-Flash-Next no-heal bench session on a RunPod A100-80GB.
# Phases are flag-gated so a rerun resumes. Any failure writes FAILED.flag.
# Layout: /workspace/{src,builds,llama.cpp,bench,results}
set -uo pipefail
W=/workspace
LOG=$W/run.log
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $LOG; }
fail(){ log "FAILED in phase $1"; echo "$1" > $W/FAILED.flag; exit 1; }
done_flag(){ [ -f $W/flags/$1.done ]; }
mark(){ mkdir -p $W/flags; touch $W/flags/$1.done; }
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH=/usr/local/cuda/bin:$PATH
cd $W

if ! done_flag env; then
  log "=== phase env ==="
  pip install -q --break-system-packages -U gguf numpy "huggingface_hub[cli,hf_transfer]" 2>&1 | tail -1
  apt-get install -y -qq psmisc cmake ccache > /dev/null 2>&1 || true
  python3 -c "import gguf, numpy" || fail env
  mark env
fi

if ! done_flag llama; then
  log "=== phase llama.cpp b10673 CUDA build ==="
  [ -d llama.cpp ] || git clone --depth 1 --branch b10673 https://github.com/ggml-org/llama.cpp 2>&1 | tail -1
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 || fail llama-cmake
  cmake --build llama.cpp/build -j "$(nproc)" --target llama-server llama-perplexity 2>&1 | tail -2 || fail llama-build
  llama.cpp/build/bin/llama-server --version 2>&1 | head -1 | tee -a $LOG
  mark llama
fi

if ! done_flag download; then
  log "=== phase download UD-Q3_K_XL ==="
  hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q3_K_XL/*" --local-dir $W/src 2>&1 | tail -1 || fail download
  mark download
fi

SRC=$W/src/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf
mkdir -p $W/builds

surgery(){ # name args...
  local name=$1; shift
  if ! done_flag "build-$name"; then
    log "=== phase build $name ==="
    python3 $W/bench/surgery_qwen38.py "$SRC" "$W/builds/$name.gguf" "$@" 2>&1 | tail -2 | tee -a $LOG || fail "build-$name"
    mark "build-$name"
  fi
}
surgery qwen38-ple0      --ple-keep 0
surgery qwen38-keep1     --ple-keep 1
surgery qwen38-k4cut     --ple-keep 4 --cut-layers
surgery qwen38-k2cut     --ple-keep 2 --cut-layers

if ! done_flag bench; then
  log "=== phase bench (5 configs) ==="
  cd $W/bench
  python3 run_bench.py --runid pod20260828 \
    --models qwen38-base,qwen38-ple0,qwen38-keep1,qwen38-k4cut,qwen38-k2cut \
    --suites tools,gsm8k,mmlu,perplexity 2>&1 | tee -a $LOG || fail bench
  cd $W
  mark bench
fi

log "=== packaging results ==="
tar czf $W/results.tgz -C $W/bench results
log "=== all phases complete ==="
touch $W/DONE.flag
