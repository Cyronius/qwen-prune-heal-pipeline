"""Speed-only probe: healed-C vs base-q4km, no MTP on either, 20 reps.
Reuses run_bench's server lifecycle and probe logic; writes nowhere near results/."""
import json, sys
sys.path.insert(0, r"C:/code/model-shrink-ideas/bench")
from run_bench import start_server, stop_server, run_speed_probe

registry = json.load(open(r"C:/code/model-shrink-ideas/bench/registry.json"))
out = {}
for name in ["base-q4km", "healed-C"]:
    cfg = registry["models"][name]
    print(f"=== {name} ===", flush=True)
    proc = start_server(registry["server_bin"], cfg["gguf"], cfg.get("extra_args", []))
    try:
        out[name] = run_speed_probe(reps=20)
    finally:
        stop_server(proc)
    print(json.dumps(out[name], indent=2), flush=True)
json.dump(out, open(r"C:/code/model-shrink-ideas/heal-artifacts/speed-test.json", "w"), indent=2)
print("DONE")
