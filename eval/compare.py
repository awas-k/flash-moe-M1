#!/usr/bin/env python3
"""Summarise eval results across K values.

Reads eval/results/k*.json written by run_eval.py and prints:
  - pass rate per K, overall and per category
  - throughput and degeneracy metrics per K
  - which individual prompts regress as K drops
  - text agreement with the highest K present (the model's trained routing)
"""
import argparse
import glob
import json
import os
import re


def load(d):
    runs = {}
    for p in sorted(glob.glob(os.path.join(d, "k*.json"))):
        m = re.match(r"k(\d+)", os.path.basename(p))
        if not m:
            continue
        rows = json.load(open(p, encoding="utf-8"))
        runs[int(m.group(1))] = {r["id"]: r for r in rows}
    return runs


def bar(frac, width=18):
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    runs = load(args.dir)
    if not runs:
        print(f"no results in {args.dir} — run eval/sweep.sh first")
        return 1

    ks = sorted(runs)
    ids = sorted({i for r in runs.values() for i in r})
    cats = sorted({r[i]["cat"] for r in runs.values() for i in r if i in r})
    ref = max(ks)

    print("=" * 74)
    print(f"flash-moe quality sweep — {len(ids)} prompts, K = {', '.join(map(str, ks))}")
    print("=" * 74)

    # ---- overall ----
    print("\nOverall pass rate")
    for k in ks:
        rows = [r for r in runs[k].values()]
        p = sum(1 for r in rows if r["ok"])
        n = len(rows)
        tag = "  (reference — trained top_k)" if k == ref else ""
        print(f"  K={k}  {bar(p/n)} {p:>2}/{n}  {100*p/n:5.1f}%{tag}")

    # ---- per category ----
    print("\nPass rate by category")
    head = "  " + "category".ljust(14) + "".join(f"  K={k}".rjust(7) for k in ks)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for c in cats:
        cells = []
        for k in ks:
            rows = [r for r in runs[k].values() if r["cat"] == c]
            cells.append(f"{sum(1 for r in rows if r['ok'])}/{len(rows)}".rjust(7))
        print("  " + c.ljust(14) + "".join(cells))

    # ---- throughput + degeneracy ----
    print("\nThroughput and degeneracy")
    print("  K    mean tok/s   mean TTFT   cap hits   U+FFFD   mean rep")
    for k in ks:
        rows = list(runs[k].values())
        ts = [r["tok_s"] for r in rows if r.get("tok_s")]
        tt = [r["ttft_s"] for r in rows if r.get("ttft_s")]
        print(f"  {k:<4} {sum(ts)/len(ts) if ts else 0:>10.2f} "
              f"{sum(tt)/len(tt) if tt else 0:>11.2f}s "
              f"{sum(1 for r in rows if r.get('hit_cap')):>10} "
              f"{sum(r.get('replacement_chars', 0) for r in rows):>8} "
              f"{sum(r.get('rep_rate', 0) for r in rows)/len(rows):>10.3f}")

    # ---- truncation check: must come before any quality claim ----
    trunc = {k: [p for p, r in runs[k].items() if r.get("status") == "truncated"]
             for k in ks}
    if any(trunc.values()):
        print("\n" + "!" * 74)
        print("INVALID RUN — some items never produced an answer: the token cap hit")
        print("while the model was still inside <think>. Those are budget failures, not")
        print("quality failures, and any pass rate below is meaningless. Raise max_tokens")
        print("in prompts.jsonl or lower the server's --think-budget, then re-run.")
        for k in ks:
            if trunc[k]:
                print(f"  K={k}: {len(trunc[k])} truncated — {', '.join(trunc[k])}")
        print("!" * 74)

    # ---- regressions vs reference ----
    print(f"\nPrompts that pass at K={ref} but fail at a lower K")
    any_reg = False
    for pid in ids:
        if pid not in runs[ref] or not runs[ref][pid]["ok"]:
            continue
        bad = [k for k in ks if k != ref and pid in runs[k]
               and not runs[k][pid]["ok"]
               and runs[k][pid].get("status") != "truncated"]
        if bad:
            any_reg = True
            cat = runs[ref][pid]["cat"]
            print(f"  {pid:<10} ({cat:<12}) fails at K={','.join(map(str, bad))}")
            for k in bad:
                print(f"       K={k}: {runs[k][pid]['why'][:66]}")
                print(f"            got {runs[k][pid]['answer'][:66]!r}")
    if not any_reg:
        print("  none")

    # ---- items the reference itself fails ----
    ref_fail = [p for p in ids if p in runs[ref] and not runs[ref][p]["ok"]]
    if ref_fail:
        print(f"\nAlso failing at K={ref} (model limitation, not a routing cost)")
        for pid in ref_fail:
            print(f"  {pid:<10} {runs[ref][pid]['why'][:60]}")

    # ---- agreement with reference text ----
    print(f"\nExact-text agreement with K={ref}")
    for k in ks:
        if k == ref:
            continue
        both = [p for p in ids if p in runs[k] and p in runs[ref]]
        same = sum(1 for p in both
                   if runs[k][p]["answer"].strip() == runs[ref][p]["answer"].strip())
        print(f"  K={k}  {same:>2}/{len(both)} identical  ({100*same/len(both):.0f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
