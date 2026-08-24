"""Schemas and prompts from the ToolACE dataset.

Why this exists: the hand-written catalog in tools_train.py has 72 tools, and at 7,000
agentic prompts each schema is shown about 280 times. The benchmark scores transfer to
seven schemas the model has never seen, so training on a small closed set risks teaching
"these 72 tools" rather than "how to read a function definition".

ToolACE carries 16,072 distinct tool schemas. Mixing it in drops the repetition per
schema by two orders of magnitude.

  source:  https://huggingface.co/datasets/Team-ACE/ToolACE
  licence: Apache 2.0
  access:  ungated, one 37 MB data.json

WHAT WE TAKE AND WHAT WE DO NOT
We take the tool schemas and the user turns. We discard every assistant turn.

Two reasons. Their assistant format is a bespoke DSL -- `[Func(arg="value")]` -- not the
OpenAI-style JSON tool calls our chat template emits. And the whole point of this
pipeline is to distil OUR teacher's behaviour, not another model's. Their assistant
turns are used only as a hint for labelling a record positive or negative.

THREE FORMAT QUIRKS, ALL HANDLED HERE
1. Half the tool names contain spaces ("Market Trends API"), which is not a legal
   function name. Names are slugged.
2. Parameter objects are typed "dict" rather than "object".
3. Each tool carries a stray top-level "required": null alongside its real parameters.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = "Team-ACE/ToolACE"
FILENAME = "data.json"

# Tools whose slugged name matches one the benchmark scores are dropped outright.
# ToolACE really does contain create_reminder, send_email and a Get Weather variant,
# so this filter is load-bearing rather than defensive.
from tools_train import EVAL_TOOL_NAMES  # noqa: E402

_TOOLS_RE = re.compile(
    r"Here is a list of functions in JSON format that you can invoke:\s*(\[.*?\])\.?\s*\n",
    re.S)
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def download(cache_dir=None):
    """Fetch data.json, using the normal Hugging Face cache unless told otherwise."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, FILENAME, repo_type="dataset",
                                local_dir=str(cache_dir) if cache_dir else None))


def slug(name):
    out = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    return out or "tool"


def convert_params(params):
    """ToolACE types objects as "dict". Rewrite to JSON Schema's "object", recursively."""
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}, "required": []}
    out = dict(params)
    if out.get("type") == "dict":
        out["type"] = "object"
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: convert_params(v) if isinstance(v, dict) and
                             v.get("type") in ("dict", "object", "array") else v
                             for k, v in props.items()}
    items = out.get("items")
    if isinstance(items, dict):
        out["items"] = convert_params(items)
    out.setdefault("properties", {})
    req = out.get("required")
    out["required"] = [r for r in req if isinstance(r, str)] if isinstance(req, list) else []
    return out


def to_openai_schema(tool):
    """One ToolACE tool -> an OpenAI-style function definition, or None if unusable."""
    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    slugged = slug(name)
    if slugged in EVAL_TOOL_NAMES:
        return None  # never train on a schema the benchmark scores
    return {"type": "function", "function": {
        "name": slugged,
        "description": (tool.get("description") or "").strip(),
        "parameters": convert_params(tool.get("parameters"))}}


def _has_call(text):
    """ToolACE marks a tool call with its [Func(arg=...)] DSL. Used only for labelling."""
    return bool(re.search(r"\[[A-Za-z_][^\]]*\(", text or ""))


def load_prompts(path=None, limit=None, max_tools=9, cache_dir=None):
    """Yield prompt rows shaped like the ones make_prompts.py produces.

    Each row is {slice, category, tools, messages} where messages end on a user turn,
    ready for gen_teacher.py --mode generate.
    """
    path = Path(path) if path else download(cache_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows, unparsed, empty = [], 0, 0

    for record in data:
        m = _TOOLS_RE.search(record.get("system", ""))
        if not m:
            unparsed += 1
            continue
        try:
            raw_tools = json.loads(m.group(1))
        except json.JSONDecodeError:
            unparsed += 1
            continue

        schemas, seen = [], set()
        for tool in raw_tools:
            s = to_openai_schema(tool)
            if s and s["function"]["name"] not in seen:
                seen.add(s["function"]["name"])
                schemas.append(s)
        if not schemas:
            empty += 1
            continue
        schemas = schemas[:max_tools]

        # Keep the leading turns up to and including the first user message. Later
        # turns depend on tool RESULTS we have not executed, so replaying them would
        # ask the teacher to continue a conversation that never happened.
        messages, first_user = [], None
        for turn in record.get("conversations", []):
            role = {"user": "user", "assistant": "assistant", "tool": "tool"}.get(turn.get("from"))
            if role == "user":
                first_user = turn.get("value")
                messages.append({"role": "user", "content": first_user})
                break
        if not first_user:
            empty += 1
            continue

        replies = [t.get("value", "") for t in record.get("conversations", [])
                   if t.get("from") == "assistant"]
        category = "toolace_positive" if any(_has_call(r) for r in replies) else "toolace_negative"

        rows.append({"slice": "agentic", "category": category,
                     "tools": schemas, "messages": messages})
        if limit and len(rows) >= limit:
            break

    return rows, {"unparsed": unparsed, "no_usable_tools": empty, "kept": len(rows)}


def schema_stats(rows):
    names = {t["function"]["name"] for r in rows for t in r["tools"]}
    impressions = sum(len(r["tools"]) for r in rows)
    return {"rows": len(rows), "distinct_schemas": len(names),
            "schema_impressions": impressions,
            "impressions_per_schema": round(impressions / max(len(names), 1), 2)}


if __name__ == "__main__":
    import sys

    rows, stats = load_prompts(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print("parse:", stats)
    print("schemas:", schema_stats(rows))
    import collections
    print("categories:", dict(collections.Counter(r["category"] for r in rows)))
    for r in rows[:3]:
        print(f"\ntools({len(r['tools'])}): {[t['function']['name'] for t in r['tools']]}")
        print(f"  user: {r['messages'][-1]['content'][:150]}")
