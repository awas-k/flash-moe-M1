#!/usr/bin/env python3
"""Re-apply the current graders to stored results, without re-running the model.

run_eval.py saves each item's raw response, so a grader fix (a too-narrow regex,
a miscalibrated length floor) can be re-scored offline in a second instead of
costing a full sweep. Generation is the expensive part; grading is not.

    python3 eval/regrade.py            # re-score eval/results in place
    python3 eval/regrade.py --dry-run  # show what would change
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import grade, strip_think, repetition_rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = {}
    for line in open(args.prompts, encoding="utf-8"):
        if line.strip():
            it = json.loads(line)
            specs[it["id"]] = it["grade"]

    total_changed = 0
    for path in sorted(glob.glob(os.path.join(args.dir, "k*.json"))):
        rows = json.load(open(path, encoding="utf-8"))
        changed = []
        for r in rows:
            if r["id"] not in specs:
                continue  # prompt removed from the set since this run
            answer, cut = strip_think(r.get("raw", ""))
            if cut:
                status, ok, why = "truncated", False, "cap hit while still inside <think>; never answered"
            else:
                ok, why = grade(answer, specs[r["id"]])
                status = "pass" if ok else "fail"
            if ok != r["ok"] or status != r.get("status"):
                changed.append((r["id"], r["ok"], ok, why))
            r.update(ok=ok, status=status, why=why, answer=answer,
                     rep_rate=repetition_rate(answer))
        if changed:
            total_changed += len(changed)
            print(f"{os.path.basename(path)}: {len(changed)} changed")
            for pid, was, now, why in changed:
                arrow = "FAIL->PASS" if now else "PASS->FAIL"
                print(f"    {pid:<10} {arrow}  {why[:56]}")
        if not args.dry_run:
            json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if not total_changed:
        print("no verdicts changed")
    elif args.dry_run:
        print(f"\n{total_changed} verdict(s) would change (dry run, nothing written)")
    else:
        print(f"\n{total_changed} verdict(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
