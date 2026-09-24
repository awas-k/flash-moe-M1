#!/usr/bin/env python3
"""Score flash-moe output quality against a fixed prompt set.

Sampling in infer.m is greedy argmax, so output is deterministic: a given
(K, prompt) always produces the same text. That makes a fixed prompt set a
real regression gate rather than a sample.

Each prompt is sent WITHOUT session_id, so every item starts from the cached
system-prompt snapshot and items cannot contaminate each other.

Usage:
    python3 eval/run_eval.py --port 8000 --k 4 --out eval/results/
"""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", re.S)
INT_RE = re.compile(r"-?\d+")


# ---------- response cleanup ----------

def strip_think(text):
    """Reasoning streams inline as content; it is not part of the answer.

    Returns (answer, truncated_in_think). The flag matters: if the token cap hit
    while still inside <think>, the model never got to answer, and that is a
    budget problem, NOT a wrong answer. Counting those as content failures
    silently turns the whole eval into a measure of reasoning length.
    """
    out = THINK_RE.sub("", text)
    truncated = "<think>" in out
    if truncated:
        out = out.split("<think>")[0]
    return out.strip(), truncated


# ---------- graders ----------
# A prompt passes only if EVERY grader key it declares passes.

def g_int_eq(ans, want):
    """First or last integer matches. Tolerates 'The answer is 391.'"""
    found = INT_RE.findall(ans.replace(",", ""))
    if not found:
        return False, "no integer in answer"
    if int(found[0]) == want or int(found[-1]) == want:
        return True, ""
    return False, f"ints {found[:3]}..{found[-1:]} != {want}"


def g_any_of(ans, needles):
    low = ans.lower()
    if any(n.lower() in low for n in needles):
        return True, ""
    return False, f"none of {needles} present"


def g_none_of(ans, needles):
    low = ans.lower()
    hit = [n for n in needles if n.lower() in low]
    if hit:
        return False, f"forbidden {hit} present"
    return True, ""


def g_regex(ans, pattern):
    if re.search(pattern, ans, re.M):
        return True, ""
    return False, f"no match for /{pattern}/"


def g_json_eq(ans, want):
    body = FENCE_RE.sub("", ans).strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end < start:
        return False, "no JSON object found"
    try:
        got = json.loads(body[start:end + 1])
    except Exception as e:
        return False, f"unparseable JSON ({e})"
    for k, v in want.items():
        if k not in got:
            return False, f"missing key {k!r}"
        if str(got[k]).strip().lower() != str(v).strip().lower():
            return False, f"{k}={got[k]!r} != {v!r}"
    return True, ""


def g_csv_len(ans, want):
    body = FENCE_RE.sub("", ans).strip().rstrip(".")
    items = [p.strip() for p in body.split(",") if p.strip()]
    if len(items) != want:
        return False, f"{len(items)} items, want {want}"
    # Reject smuggled prose ("Here are three fruits: a, b, c")
    if any(len(i.split()) > 3 for i in items):
        return False, f"items look like prose: {items}"
    return True, ""



def g_max_chars(ans, limit):
    """Enforce 'answer with only X'. Without this, a degenerate 500-token
    ramble that happens to contain the right substring scores as a pass."""
    if len(ans) <= limit:
        return True, ""
    return False, f"{len(ans)} chars > {limit} (answer not concise)"


def g_min_chars(ans, floor):
    """Long-form items must actually elaborate, not answer in one word."""
    if len(ans) >= floor:
        return True, ""
    return False, f"{len(ans)} chars < {floor} (too short to judge)"


def g_distinct_orgs(ans, need):
    """Guard against 'three models' that are three variants from one lab."""
    orgs = ["mistral", "google", "xai", "alibaba", "qwen", "deepseek",
            "meta", "openai", "microsoft", "databricks", "snowflake", "ai21"]
    low = ans.lower()
    found = {o for o in orgs if o in low}
    # Qwen is Alibaba's; do not let the pair count twice.
    if "qwen" in found and "alibaba" in found:
        found.discard("qwen")
    if len(found) >= need:
        return True, ""
    return False, f"only {len(found)} distinct org(s) named {sorted(found)}, need {need}"


GRADERS = {
    "int_eq": g_int_eq, "any_of": g_any_of, "none_of": g_none_of,
    "regex": g_regex, "json_eq": g_json_eq, "csv_len": g_csv_len,
    "max_chars": g_max_chars, "min_chars": g_min_chars,
    "distinct_orgs": g_distinct_orgs,
}


def grade(answer, spec):
    reasons = []
    for key, want in spec.items():
        fn = GRADERS.get(key)
        if fn is None:
            reasons.append(f"unknown grader {key!r}")
            continue
        ok, why = fn(answer, want)
        if not ok:
            reasons.append(why)
    return (not reasons), "; ".join(reasons)


# ---------- degeneracy metrics (need no ground truth) ----------

def repetition_rate(text, n=4):
    toks = text.split()
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


# ---------- server I/O ----------

def ask(port, prompt, max_tokens, timeout=900):
    """One turn, no session_id, SSE parsed. Returns (text, stats)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.time()
    first = None
    tokens = 0
    parts = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith(":") or not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                saw_done = True
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            piece = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if not piece:
                continue
            if first is None:
                first = time.time()
            tokens += 1
            parts.append(piece)
    end = time.time()
    text = "".join(parts)
    return text, {
        "tokens": tokens,
        "ttft_s": round((first - t0), 3) if first else None,
        "tok_s": round(tokens / (end - first), 2) if first and end > first else None,
        "hit_cap": tokens >= max_tokens,
        "saw_done": saw_done,
        # U+FFFD counts the sse_send_delta UTF-8 splitting bug (infer.m:6355)
        "replacement_chars": text.count("�"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, required=True, help="K the server was started with (recorded, not sent)")
    ap.add_argument("--prompts", default=str(pathlib.Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results"))
    ap.add_argument("--tag", default="", help="optional label for this run")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, it in enumerate(items, 1):
        print(f"[{i:2d}/{len(items)}] K={args.k} {it['id']:<9} ", end="", flush=True)
        try:
            raw, stats = ask(args.port, it["prompt"], it.get("max_tokens", 128))
        except Exception as e:
            print(f"REQUEST FAILED: {e}")
            rows.append({**it, "k": args.k, "ok": False, "why": f"request failed: {e}",
                         "answer": "", "raw": "", "tokens": 0})
            continue
        answer, cut_in_think = strip_think(raw)
        if cut_in_think:
            status, ok, why = "truncated", False, "cap hit while still inside <think>; never answered"
        else:
            ok, why = grade(answer, it["grade"])
            status = "pass" if ok else "fail"
        rows.append({
            "id": it["id"], "cat": it["cat"], "k": args.k, "ok": ok,
            "status": status, "why": why,
            "answer": answer, "raw": raw, "rep_rate": repetition_rate(answer), **stats,
        })
        flag = "" if not stats["replacement_chars"] else f" U+FFFD×{stats['replacement_chars']}"
        label = {"pass": "PASS", "fail": "FAIL", "truncated": "TRUNC"}[status]
        print(f"{label}  {stats['tokens']:>4}tok "
              f"{stats['tok_s'] or 0:>5.2f}tok/s{flag}"
              f"{'' if ok else '  <- ' + why[:60]}")

    stem = f"k{args.k}" + (f"-{args.tag}" if args.tag else "")
    (outdir / f"{stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = sum(1 for r in rows if r["ok"])
    trunc = sum(1 for r in rows if r.get("status") == "truncated")
    print(f"\nK={args.k}: {passed}/{len(rows)} passed  ->  {outdir / (stem + '.json')}")
    if trunc:
        print(f"  WARNING: {trunc} item(s) never answered — the cap hit during <think>. "
              f"Raise max_tokens or lower the server's --think-budget; "
              f"these are NOT quality failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
