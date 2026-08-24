import json, sys
sys.path.insert(0, r"C:/code/model-shrink-ideas/bench")
import run_bench
from run_bench import start_server, stop_server, run_tools_suite
from pathlib import Path
BIN = r"C:/code/model-shrink-ideas/bench/tools/llama-current/llama-server.exe"
GGUF = r"C:/code/model-shrink-ideas/qwen36-healed-C-mtp-Q4KM.gguf"
work = Path(r"C:/code/model-shrink-ideas/bench/results/mtp-confirm-work")
work.mkdir(parents=True, exist_ok=True)
proc = start_server(BIN, GGUF, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])
try:
    results = run_tools_suite(work, None)
finally:
    stop_server(proc)
ok = sum(1 for r in results if r.get("correct"))
print(f"MTP-CONFIRM tools: {ok}/{len(results)}")
import collections
cats = collections.defaultdict(lambda: [0,0])
for r in results:
    cats[r["category"]][0]+=1; cats[r["category"]][1]+=bool(r.get("correct"))
for c,(n,k) in sorted(cats.items()): print(f"  {c}: {k}/{n}")
