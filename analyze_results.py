#!/usr/bin/env python3
"""
analyze_results.py — Analyze bench_matrix.sh experiment results.

Reads results.tsv and prints summary stats grouped by tag, K value, etc.
Designed for quick comparison during autoresearch iteration.

Usage:
    python analyze_results.py                     # full summary
    python analyze_results.py --tag baseline      # filter by tag
    python analyze_results.py --compare a b       # compare two tags
    python analyze_results.py --best              # show best config per tag
"""

import argparse
import csv
import sys
from collections import defaultdict


def load_results(path="results.tsv"):
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    row["tok_s"] = float(row.get("tok_s", 0))
                    row["ttft_s"] = float(row.get("ttft_s", 0))
                    row["tokens_out"] = int(row.get("tokens_out", 0))
                    row["K"] = int(row.get("K", 0))
                    row["run"] = int(row.get("run", 0))
                except (ValueError, TypeError):
                    continue
                rows.append(row)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run bench_matrix.sh first.")
        sys.exit(1)
    return rows


def stats(values):
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "n": 0}
    n = len(values)
    mean = sum(values) / n
    return {"mean": mean, "min": min(values), "max": max(values), "n": n}


def print_summary(rows, title="All results"):
    if not rows:
        print("No matching results.")
        return

    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    # Group by (tag, K)
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("tag", "?"), r["K"])].append(r)

    print(f"\n{'tag':<20} {'K':>3} {'runs':>5} {'tok/s':>10} {'ttft':>10} {'quality':>8}")
    print(f"{'-'*20} {'-'*3} {'-'*5} {'-'*10} {'-'*10} {'-'*8}")

    for (tag, k) in sorted(groups.keys()):
        rr = groups[(tag, k)]
        tok_vals = [r["tok_s"] for r in rr if r.get("quality") != "crash"]
        ttft_vals = [r["ttft_s"] for r in rr if r.get("quality") != "crash"]
        ts = stats(tok_vals)
        tt = stats(ttft_vals)
        quals = [r.get("quality", "?") for r in rr]
        pass_rate = sum(1 for q in quals if q == "pass") / len(quals) * 100

        tok_str = f"{ts['mean']:.2f} ({ts['min']:.1f}-{ts['max']:.1f})" if ts["n"] > 1 else f"{ts['mean']:.2f}"
        ttft_str = f"{tt['mean']:.2f}" if tt["n"] > 0 else "n/a"
        qual_str = f"{pass_rate:.0f}%" if len(quals) > 0 else "?"

        print(f"{tag:<20} {k:>3} {ts['n']:>5} {tok_str:>10} {ttft_str:>10} {qual_str:>8}")


def print_comparison(rows, tag_a, tag_b):
    a_rows = [r for r in rows if r.get("tag") == tag_a]
    b_rows = [r for r in rows if r.get("tag") == tag_b]

    print(f"\n{'=' * 70}")
    print(f"  Comparison: {tag_a} vs {tag_b}")
    print(f"{'=' * 70}")

    k_values = sorted(set(r["K"] for r in a_rows + b_rows))

    print(f"\n{'K':>3} {'':>4} {tag_a:>12} {tag_b:>12} {'delta':>10} {'change':>8}")
    print(f"{'-'*3} {'-'*4} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    for k in k_values:
        a_tok = [r["tok_s"] for r in a_rows if r["K"] == k and r.get("quality") != "crash"]
        b_tok = [r["tok_s"] for r in b_rows if r["K"] == k and r.get("quality") != "crash"]

        if a_tok and b_tok:
            a_mean = sum(a_tok) / len(a_tok)
            b_mean = sum(b_tok) / len(b_tok)
            delta = b_mean - a_mean
            pct = (delta / a_mean * 100) if a_mean > 0 else 0
            sign = "+" if delta >= 0 else ""
            emoji = "faster" if delta > 0.1 else ("slower" if delta < -0.1 else "same")
            print(f"{k:>3} tok/s {a_mean:>10.2f}  {b_mean:>10.2f}  {sign}{delta:>8.2f}  {sign}{pct:.1f}% {emoji}")
        elif a_tok:
            print(f"{k:>3} tok/s {sum(a_tok)/len(a_tok):>10.2f}  {'n/a':>12}")
        elif b_tok:
            print(f"{k:>3} tok/s {'n/a':>12}  {sum(b_tok)/len(b_tok):>10.2f}")


def print_best(rows):
    print(f"\n{'=' * 70}")
    print(f"  Best configuration per tag")
    print(f"{'=' * 70}")

    tags = sorted(set(r.get("tag", "?") for r in rows))

    print(f"\n{'tag':<20} {'best K':>6} {'tok/s':>8} {'ttft':>8}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*8}")

    for tag in tags:
        tag_rows = [r for r in rows if r.get("tag") == tag and r.get("quality") != "crash"]
        if not tag_rows:
            continue

        groups = defaultdict(list)
        for r in tag_rows:
            groups[r["K"]].append(r["tok_s"])

        best_k = max(groups.keys(), key=lambda k: sum(groups[k]) / len(groups[k]))
        best_tok = sum(groups[best_k]) / len(groups[best_k])
        best_ttft = sum(r["ttft_s"] for r in tag_rows if r["K"] == best_k) / len(groups[best_k])

        print(f"{tag:<20} {best_k:>6} {best_tok:>8.2f} {best_ttft:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze bench_matrix results")
    parser.add_argument("--file", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--tag", default=None, help="Filter by tag")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="Compare two tags")
    parser.add_argument("--best", action="store_true", help="Show best config per tag")
    args = parser.parse_args()

    rows = load_results(args.file)
    print(f"Loaded {len(rows)} results from {args.file}")

    if args.compare:
        print_comparison(rows, args.compare[0], args.compare[1])
    elif args.best:
        print_best(rows)
    elif args.tag:
        filtered = [r for r in rows if r.get("tag") == args.tag]
        print_summary(filtered, title=f"Tag: {args.tag}")
    else:
        print_summary(rows)
        if len(set(r.get("tag") for r in rows)) > 1:
            print_best(rows)


if __name__ == "__main__":
    main()
