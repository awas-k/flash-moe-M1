# Q4 Optimization Experiments (M3 Baseline + M4 Port + M1 Pro Port)

## Summary

This document tracks four phases:

1. Original M3 Max optimization pass (48GB), culminating at 4.4 tok/s.
2. M4 16GB port + optimization pass, culminating at 11.5 tok/s and 2.5s TTFT.
3. M1 Pro 16GB port + optimization pass. Best reproducible result is 11.53 tok/s / 1.94s TTFT
   at K=4 (`8a217dd`); the widely-quoted 11.91 came from an uncommitted tree.
4. Quality sweep across K (2026-09-24) — the first measurement of what reducing K costs.
   K=4 matches trained top-8 routing at 1.44× the speed; **K=3 is degraded and should not be
   used**. See [K Quality Sweep](#3-k-quality-sweep-2026-09-24).

## Results Comparison

| Phase | Hardware | K | Sustained tok/s | TTFT | Outcome |
|---|---|---:|---:|---:|---|
| Original baseline | M3 Max 48GB | 4 | 4.4 | ~5.6s | First production-quality release |
| M4 port | M4 16GB | 6 | **11.5** | **2.5s** | 2.6× faster on lower-cost hardware |
| M1 Pro port | M1 Pro 16GB | 4 | **11.91** | **1.71s** | See note — not a like-for-like win over M4 |

> The M1 Pro row is K=4 and the M4 row is K=6, so they are not comparable. At **matched K=6**
> the M1 Pro does 9.66 tok/s against the M4's 11.5 — **16% slower**, not faster. Earlier
> revisions described this as "exceeds M4 reference on older silicon"; that compared the two
> at different routing widths. The 11.91 figure is also not reproducible from any commit —
> see the tg256 provenance note in the README.

## M3 Max Track (Historical)

The original optimization series established:

- Stable 4-bit quality with tool calling.
- SSD streaming as the practical path for routed experts.
- Baseline pipeline behavior and first-wave kernel tuning.

Representative kept/discarded experiments from that phase remain relevant for context, including cache strategy and prefetch behavior under Apple unified memory.

## M1 Pro 16GB Port: What Changed

The M1 Pro pass started from tayoun's M4-optimized codebase and re-tuned for M1 Pro's
different hardware characteristics: 200 GB/s memory bandwidth against the base M4's 120, and
14 GPU cores against 10. SSD read measures 5.14–5.34 GB/s here (`F_NOCACHE`, sequential,
single stream).

> Earlier revisions added "vs M4's ~17.5 GB/s" and treated the SSD as the binding constraint.
> That M4 figure was never measured and exceeds the ~8 GB/s ceiling of PCIe 4.0 ×4, so it has
> been dropped. The runtime's own `-T` breakdown puts GPU synchronisation at ~65% of token
> time and expert I/O at ~31% — see "Where the time actually goes" in the README.

### 1. `tg256` Expert Matvec Selection

tayoun's `tg128` was tuned for M4's 10-core GPU and underperformed on M1 Pro.
Removing the heuristic that selects it (`beb9e47`) falls through to the pre-existing
256-thread `matvec_v3`, which better saturates M1 Pro's 200 GB/s and its 14 GPU cores.

- CMD1 improvement: −0.106 ms/layer
- CMD2 improvement: −0.082 ms/layer
- Net throughput gain at K=4: **~+0.6 tok/s (+5.5%)** — 10.93 (`089cb24`) → 11.53 (`8a217dd`)

> No kernel was written: `beb9e47` changes 17 lines of `infer.m` and never touches
> `shaders.metal`. The earlier "+0.98 tok/s (+9%)" compared against a run measured on an
> uncommitted tree; committed-vs-committed the gain is +5.5%, and even that is confounded
> with the other changes in that commit. See the README's provenance note.

**Why not tg512?** At `in_dim=2048`, tg512 gives each thread only 4 loop iterations
before the reduction phase dominates. Combined with higher synchronization cost and
reduced occupancy (2 threadgroups/core vs 4), tg512 adds overhead without additional
bandwidth gain — 200 GB/s is already saturated at tg256. tg256 is the sweet spot for
M1 Pro's specific core count and bandwidth profile.

### 2. K Tuning for M1 Pro

The M4 production default of K=6 is suboptimal on M1 Pro due to its slower SSD.
Each additional expert adds ~1.77 MB × 40 layers of SSD I/O per token.

- Recommended: `K=4` (11.53 tok/s, 1.94s TTFT at `8a217dd`) — best quality/speed balance
- Maximum throughput: `K=3` (12.77 tok/s, 1.66s TTFT) — **not recommended**, see the quality
  sweep below; `quality=pass` in `bench.sh` is a liveness check (≥32 non-empty tokens), never
  a quality measurement, so it never supported this
- K=6 on M1 Pro: 9.66 tok/s — 16% slower than the M4 reference at the same K

Note that `K` truncates the router's trained top-8 (`num_experts_per_tok: 8` in `config.json`)
and renormalises (`infer.m:5590`). `cfg.num_experts_per_tok` is parsed but never drives
routing, so every K below 8 discards routing mass the model was trained to use. Until the
sweep below, nothing measured what that cost.

### 3. K Quality Sweep (2026-09-24)

Run with `eval/sweep.sh` over 26 scored prompts; see `eval/README.md`. Sampling is greedy
`cpu_argmax` with `temperature`/`top_p`/`seed` parsed nowhere, so output is deterministic and
any difference between two K values is attributable to K alone.

| K | pass | tok/s | cap hits | U+FFFD | repetition | agreement w/ K=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 23/26 | 11.22 | 6 | 38 | 0.041 | 54% |
| 4 | 24/26 | 10.29 | 3 | 0 | 0.009 | 58% |
| 6 | 25/26 | 8.36 | 1 | 0 | 0.000 | 65% |
| 8 | 24/26 | 7.16 | 2 | 0 | 0.000 | — (reference) |

**K=3 is measurably degraded, and the failure mode is verbosity rather than wrong facts.**
Both of its regressions are it failing to stop: a one-word translation answered in 370
characters, and a 2-3 sentence question answered in 1814 characters that leaked its own
drafting process (`*Critique 1:* A bit clunky. Let's refine. *Draft 2:*`). Four independent
metrics agree — double the cap hits, 4.5× the repetition rate, 38 U+FFFD against zero
elsewhere, and the lowest text agreement with trained routing.

**K=4, 6 and 8 are indistinguishable**: 24/25/24 of 26 are spreads of a single item. K=4
therefore matches trained top-8 routing on this set while running **1.44× faster** (10.29 vs
7.16 tok/s), which supports the existing default rather than overturning it. Note the
throughput figures are lower than the 256-token benchmark because these are 512-token runs;
the benchmark numbers come from shorter runs with an empty context.

Text agreement with K=8 rises monotonically with K (54% → 58% → 65%) even where both answers
grade correct — lower K diverges from trained routing before it starts being wrong.

One prompt (`moe-02`, Switch Transformer's expert count) fails at K=4, 6 **and** 8: the model
believes it activates two experts per token when the correct answer is one. That is a
knowledge error at trained routing, not a cost of reducing K, and `compare.py` reports such
items separately so they are not charged to a lower K.

## Current M1 Pro Production Profile

- Model: Qwen3.5-35B-A3B-4bit
- Hardware: MacBook Pro M1 Pro, 16GB unified memory
- Routing: `K=4` — matches trained top-8 routing on the 26-prompt eval at 1.44× the speed
- Performance: 11.53 tok/s sustained, 1.94s TTFT (git `8a217dd`) — best reproducible;
  ~9.5–10.5 tok/s in sustained chat as context grows
- Stability: no crashes across benchmark runs. (Previously written as "quality=pass across
  all benchmark runs", which overstated it: `bench.sh`'s `quality=pass` only asserts that
  ≥32 non-empty tokens were produced. Quality is measured by `eval/`, not by `bench.sh`.)

## M4 16GB Port: from Fork 

The M4 pass focused on end-to-end token latency rather than isolated microbench gains.

### 1. `tg128` Matvec Kernels

- Adopted `tg128` kernel variants in hot matvec paths.
- Increased effective threadgroup utilization on M4.
- Reduced matvec-dominant per-layer latency in routed expert compute.

### 2. Encoder Coalescing

- Coalesced encode/prefill work to reduce synchronization and launch overhead.
- Lowered non-matvec overhead in prompt processing and token-step setup.
- Major contributor to TTFT reduction.

### 3. Kernel Fusion

- Fused latency-critical adjacent operations in hot paths.
- Reduced intermediate memory traffic and CPU-GPU round trips.
- Improved sustained decode throughput in steady state.

### 4. K Became Runtime-Configurable

- K was previously tuned around fixed M3 assumptions.
- Runtime now supports configurable `--k`; current M4 production profile uses `K=6`.
- This enables hardware-specific tradeoff tuning without code changes.

## Current M4 Production Profile

- Model: Qwen3.5-35B-A3B-4bit
- Hardware: Mac mini M4, 16GB unified memory
- Routing: `K=6`
- Performance: 11.5 tok/s sustained, 2.5s TTFT
- Stability: zero crashes in benchmark and tool-calling use


## Lessons

- M-series generation changes can overturn previous kernel assumptions — tg128 (M4-optimal) hurt M1 Pro; tg256 recovered and exceeded M4 throughput.
- SSD speed dominates at K>4 on 16GB machines; K must be tuned per-chip, not just per-memory-tier.
- Memory bandwidth headroom (200 GB/s on M1 Pro) can compensate for slower SSD if kernels are sized to saturate it.
- Launch/sync overhead matters at this scale; coalescing and fusion compound.
- End-to-end throughput improvements can significantly exceed per-kernel microbench deltas when multiple bottlenecks are addressed together.
- A liveness check is not a quality check. `bench.sh`'s `quality=pass` only asserts that ≥32
  non-empty tokens came back, yet it was cited for a year as evidence that lower K was
  quality-neutral. A speed knob that trades away accuracy needs a scored gate before it can be
  recommended, not a smoke test.
- Reducing K degrades fluency before it degrades facts. K=3 did not start getting answers
  wrong; it started failing to stop — rambling, repeating, leaking its drafting process. So
  watch verbosity and repetition metrics, not just correctness, when evaluating a routing
  change. Cap hits, repetition rate and output length needed no ground truth and caught it.
- Measure the eval before trusting it. The first sweep sized `max_tokens` too small, the model
  spent the whole budget inside `<think>`, and every unanswered item scored as a quality
  failure — a harness that confidently measured reasoning length while looking like a quality
  result. Truncation is now a distinct status that invalidates the whole report.
- Do not generalise from one transcript. A single chat suggested K=4 degraded technical
  terminology (rendering the MoE router as 「ガバナー」). Guards written specifically to catch
  that pass at every K, including 3. It was a one-off.
