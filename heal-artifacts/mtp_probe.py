import json, sys
sys.path.insert(0, r"C:/code/model-shrink-ideas/bench")
from run_bench import start_server, stop_server, run_speed_probe

BIN = r"C:/code/model-shrink-ideas/bench/tools/llama-current/llama-server.exe"
GGUF = r"C:/code/model-shrink-ideas/qwen36-healed-C-mtp-Q4KM.gguf"
K6 = ["--override-kv", "qwen35moe.expert_used_count=int:6"]
MTP = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
configs = {
    "k8-nospec": [],
    "k8-mtp":    MTP,
    "k6-nospec": K6,
    "k6-mtp":    K6 + MTP,
}
out = {}
for name, extra in configs.items():
    print(f"=== {name} ===", flush=True)
    proc = start_server(BIN, GGUF, extra)
    try:
        out[name] = run_speed_probe(reps=12)
    finally:
        stop_server(proc)
    print(json.dumps({k: v for k, v in out[name].items() if "tg" in k or "pp" in k}, indent=1), flush=True)
json.dump(out, open(r"C:/code/model-shrink-ideas/heal-artifacts/mtp-probe.json", "w"), indent=2)
print("PROBE-DONE")
