"""Prune benchmark harness: smarts + speed for GGUF models behind llama-server.

Per model in registry.json: start llama-server, run suites (tools / gsm8k / mmlu),
kill server, then llama-bench + llama-perplexity from official release binaries.
Deterministic scoring, no LLM judge. Resumable via per-item work files.

Usage:
  python run_bench.py                         # all models, all suites, new run id
  python run_bench.py --models base-q4km --limit 2   # harness smoke test
  python run_bench.py --runid 20260805T120000Z       # resume a run
"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

BENCH = Path(__file__).parent
PORT = 8199
BASE_URL = f"http://127.0.0.1:{PORT}"
REQUEST_TIMEOUT = 900
# tools keeps thinking on (matches real agentic traffic); gsm8k/mmlu run with
# thinking disabled so a 5-model round fits in hours, not days
CAPS = {"tools": 1024, "gsm8k": 1024, "mmlu": 256}
NO_THINK = {"gsm8k", "mmlu"}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def post_chat(payload):
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        body = json.load(r)
    body["_elapsed_s"] = time.time() - t0
    return body


# ---------------- server lifecycle ----------------

def start_server(server_bin, gguf, extra_args, _retry=True):
    subprocess.run(["powershell", "-NoProfile", "-Command", f"killport {PORT}"], capture_output=True)
    time.sleep(2)
    proc = subprocess.Popen(
        [server_bin, "-m", gguf, "--port", str(PORT), "-ngl", "99", "-c", "8192", *extra_args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode} while loading {gguf}")
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=5) as r:
                if b"ok" in r.read():
                    log("server healthy")
                    return proc
        except Exception:
            pass
        time.sleep(3)
    proc.kill()
    if _retry:
        log("server load timed out; retrying once")
        time.sleep(30)
        return start_server(server_bin, gguf, extra_args, _retry=False)
    raise RuntimeError(f"server never became healthy for {gguf}")


def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    subprocess.run(["powershell", "-NoProfile", "-Command", f"killport {PORT}"], capture_output=True)


# ---------------- scoring helpers ----------------

def norm(v):
    return str(v).strip().lower()


def num_eq(a, b):
    try:
        return abs(float(str(a).replace(",", "").replace("$", "")) - float(str(b).replace(",", "").replace("$", ""))) < 1e-4
    except (ValueError, TypeError):
        return False


def match_arg(value, matcher):
    if "eq" in matcher:
        return any(num_eq(value, alt) or norm(value) == norm(alt) for alt in matcher["eq"])
    if "has" in matcher:
        return all(norm(sub) in norm(value) for sub in matcher["has"])
    return False


def score_tool_case(case, message, finish_reason):
    exp = case["expect"]
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    looped = finish_reason == "length" and not tool_calls and not content.strip()

    if not exp["call"]:
        ok_call = not tool_calls
        contains = exp.get("content_contains")
        ok_content = True if not contains else any(norm(s) in norm(content) for s in contains)
        return {"correct": ok_call and ok_content, "called_when_shouldnt": bool(tool_calls), "looped": looped}

    if not tool_calls:
        return {"correct": False, "no_call_when_should": True, "looped": looped}
    tc = tool_calls[0]["function"]
    if tc.get("name") != exp["name"]:
        return {"correct": False, "wrong_function": tc.get("name"), "looped": looped}
    try:
        args = json.loads(tc.get("arguments") or "{}")
    except json.JSONDecodeError:
        return {"correct": False, "invalid_json": True, "looped": looped}
    expected_args = exp.get("args", {})
    allowed = set(expected_args) | set(exp.get("args_optional", []))
    if any(k not in allowed for k in args):
        return {"correct": False, "unexpected_args": sorted(set(args) - allowed), "looped": looped}
    for key, matcher in expected_args.items():
        if key not in args:
            return {"correct": False, "missing_arg": key, "looped": looped}
        if not match_arg(args[key], matcher):
            return {"correct": False, "bad_arg": {key: args[key]}, "looped": looped}
    return {"correct": True, "looped": looped}


GSM_ANS = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")
LAST_NUM = re.compile(r"(-?[\d,]+(?:\.\d+)?)(?!.*-?[\d,]+(?:\.\d+)?)", re.DOTALL)
MMLU_ANSWER = re.compile(r"answer[^ABCD]{0,20}\b([ABCD])\b", re.IGNORECASE)
MMLU_LETTER = re.compile(r"\b([ABCD])\b")


def parse_mmlu_letter(text):
    hits = MMLU_ANSWER.findall(text)
    if hits:
        return hits[-1]
    hits = MMLU_LETTER.findall(text)
    return hits[-1] if hits else None


def parse_gsm_answer(text):
    m = GSM_ANS.search(text)
    if m:
        return m.group(1)
    m = LAST_NUM.search(text)
    return m.group(1) if m else None


# ---------------- suites ----------------

def run_tools_suite(work, limit):
    catalog = json.loads((BENCH / "tools_catalog.json").read_text(encoding="utf-8"))
    cases = [json.loads(l) for l in (BENCH / "cases" / "tools.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        cases = cases[:limit]
    results = []
    for case in cases:
        f = work / f"{case['id']}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
            continue
        payload = {
            "messages": case["messages"],
            "tools": [catalog[t] for t in case["tools"]],
            "temperature": 0,
            "max_tokens": CAPS["tools"],
        }
        try:
            resp = post_chat(payload)
            choice = resp["choices"][0]
            item = {"id": case["id"], "category": case["category"],
                    **score_tool_case(case, choice["message"], choice.get("finish_reason")),
                    "elapsed_s": resp["_elapsed_s"],
                    "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
                    "timings": resp.get("timings")}
        except Exception as e:
            item = {"id": case["id"], "category": case["category"], "correct": False, "error": str(e), "looped": False}
        f.write_text(json.dumps(item), encoding="utf-8")
        results.append(item)
        log(f"  tools {case['id']}: {'OK' if item['correct'] else 'FAIL'}")
    return results


def run_qa_suite(suite, work, limit):
    data = [json.loads(l) for l in (BENCH / "data" / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        data = data[:limit]
    results = []
    for i, item_data in enumerate(data):
        f = work / f"{i:04d}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
            continue
        if suite == "gsm8k":
            prompt = (item_data["question"] +
                      "\n\nSolve step by step, then give ONLY the final numeric answer on the last line in the form: #### <number>")
        else:  # mmlu
            letters = "ABCD"
            choices = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(item_data["choices"]))
            prompt = f"{item_data['question']}\n{choices}\n\nAnswer with just the letter of the correct choice."
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": CAPS[suite]}
        if suite in NO_THINK:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = post_chat(payload)
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
            if suite == "gsm8k":
                parsed = parse_gsm_answer(content)
                correct = parsed is not None and num_eq(parsed, item_data["gold"])
            else:
                parsed = parse_mmlu_letter(content)
                correct = parsed == "ABCD"[item_data["answer"]]
            looped = finish == "length" and parsed is None
            item = {"idx": i, "correct": correct, "parsed": parsed, "gold": item_data.get("gold", item_data.get("answer")),
                    "content_tail": content[-200:],
                    "looped": looped, "elapsed_s": resp["_elapsed_s"],
                    "completion_tokens": resp.get("usage", {}).get("completion_tokens"),
                    "timings": resp.get("timings")}
        except Exception as e:
            item = {"idx": i, "correct": False, "error": str(e), "looped": False}
        f.write_text(json.dumps(item), encoding="utf-8")
        results.append(item)
        log(f"  {suite} {i}: {'OK' if item['correct'] else 'FAIL'}")
    return results


# ---------------- speed ----------------

def run_speed_probe(reps=10):
    """Repeated representative requests against the live server; timings from llama-server itself.
    Prompts vary at the FIRST token per rep so the server's prompt cache can't inflate pp numbers."""
    import statistics
    catalog = json.loads((BENCH / "tools_catalog.json").read_text(encoding="utf-8"))
    topics = ["mountain hiking", "deep sea fishing", "urban gardening", "winter camping", "desert astronomy",
              "river kayaking", "forest foraging", "coastal cycling", "cave exploring", "alpine skiing"]
    cities = ["Tokyo", "Paris", "Denver", "Oslo", "Lima", "Cairo", "Perth", "Quebec", "Hanoi", "Lagos"]
    post_chat({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, "temperature": 0})  # warmup
    samples = {"completion_pp": [], "completion_tg": [], "tool_call_pp": [], "tool_call_tg": []}
    errors = []
    for i in range(reps):
        probes = {
            "completion": {"messages": [{"role": "user", "content": f"[{i}] Write a short paragraph about {topics[i % len(topics)]}."}],
                           "max_tokens": 256, "temperature": 0},
            "tool_call": {"messages": [{"role": "user", "content": f"[{i}] What's the weather in {cities[i % len(cities)]} right now?"}],
                          "tools": [catalog["get_weather"]], "max_tokens": 256, "temperature": 0},
        }
        for label, payload in probes.items():
            try:
                t = post_chat(payload).get("timings") or {}
                if t.get("prompt_per_second"):
                    samples[f"{label}_pp"].append(t["prompt_per_second"])
                if t.get("predicted_per_second"):
                    samples[f"{label}_tg"].append(t["predicted_per_second"])
            except Exception as e:
                errors.append(f"{label}[{i}]: {e}")
    speed = {"speed_reps": reps}
    for key, vals in samples.items():
        if vals:
            speed[f"{key}_tps_mean"] = round(statistics.mean(vals), 2)
            speed[f"{key}_tps_sd"] = round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0
    if errors:
        speed["speed_errors"] = errors[:5]
    return speed


def run_perplexity(model_cfg):
    ppl_bin = BENCH / "tools" / "llama" / "llama-perplexity.exe"
    wiki = BENCH / "data" / "wikitext-2-raw" / "wiki.test.raw"
    out = subprocess.run(
        [str(ppl_bin), "-m", model_cfg["gguf"], "-f", str(wiki), "--chunks", "32", "-c", "512", "-ngl", "99"],
        capture_output=True, text=True, timeout=7200)
    m = re.search(r"Final estimate: PPL = ([\d.]+)", out.stdout + out.stderr)
    return {"ppl_wikitext2_32chunk": float(m.group(1))} if m else {"error": (out.stderr or out.stdout)[-500:]}


# ---------------- aggregation ----------------

def aggregate(results_by_suite, speed, ppl, model_name, model_cfg, runid):
    def rate(items, key="correct"):
        return round(sum(1 for r in items if r.get(key)) / len(items), 4) if items else None

    gen_items = [r for suite in results_by_suite.values() for r in suite]
    tools = results_by_suite.get("tools", [])
    by_cat = {}
    for cat in sorted({t["category"] for t in tools}):
        by_cat[cat] = rate([t for t in tools if t["category"] == cat])
    return {
        "runid": runid, "model": model_name, "gguf": model_cfg["gguf"],
        "size_gb": round(Path(model_cfg["gguf"]).stat().st_size / 1e9, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tools_acc": rate(tools), "tools_by_category": by_cat,
        "gsm8k_acc": rate(results_by_suite.get("gsm8k", [])),
        "mmlu_acc": rate(results_by_suite.get("mmlu", [])),
        "loop_rate": rate(gen_items, "looped"),
        **speed, **ppl,
        "n": {k: len(v) for k, v in results_by_suite.items()},
    }


def write_summary(results_dir):
    latest = {}
    for f in sorted(results_dir.glob("*_result_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        latest[r["model"]] = r
    cols = ["model", "size_gb", "tools_acc", "gsm8k_acc", "mmlu_acc", "loop_rate",
            "ppl_wikitext2_32chunk", "completion_pp_tps_mean", "completion_tg_tps_mean",
            "completion_tg_tps_sd", "tool_call_tg_tps_mean"]
    lines = ["# Prune benchmark summary", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for name in sorted(latest):
        r = latest[name]
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--suites", default="tools,gsm8k,mmlu,speed,perplexity")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--runid", default=None)
    args = ap.parse_args()

    registry = json.loads((BENCH / "registry.json").read_text(encoding="utf-8"))
    runid = args.runid or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suites = args.suites.split(",")
    models = args.models.split(",") if args.models else list(registry["models"])
    results_dir = BENCH / "results"
    results_dir.mkdir(exist_ok=True)
    log(f"run {runid}: models={models} suites={suites} limit={args.limit}")

    for name in models:
        cfg = registry["models"][name]
        if not Path(cfg["gguf"]).exists():
            log(f"SKIP {name}: missing {cfg['gguf']}")
            continue
        log(f"=== {name} ===")
        results_by_suite, proc, speed = {}, None, {}
        gen_suites = [s for s in suites if s in ("tools", "gsm8k", "mmlu")]
        try:
            if gen_suites or "speed" in suites:
                proc = start_server(registry["server_bin"], cfg["gguf"], cfg.get("extra_args", []))
                if "speed" in suites:
                    speed = run_speed_probe()
                for suite in gen_suites:
                    work = results_dir / f"{runid}_work" / name / suite
                    work.mkdir(parents=True, exist_ok=True)
                    if suite == "tools":
                        results_by_suite[suite] = run_tools_suite(work, args.limit)
                    else:
                        results_by_suite[suite] = run_qa_suite(suite, work, args.limit)
        finally:
            stop_server(proc)
        ppl = run_perplexity(cfg) if "perplexity" in suites else {}
        result = aggregate(results_by_suite, speed, ppl, name, cfg, runid)
        out = results_dir / f"{runid}_result_{name}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"{name}: tools={result['tools_acc']} gsm8k={result['gsm8k_acc']} mmlu={result['mmlu_acc']} "
            f"loop={result['loop_rate']} tg128={result.get('tg128_ts')}")
    write_summary(results_dir)
    log(f"summary written to {results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
