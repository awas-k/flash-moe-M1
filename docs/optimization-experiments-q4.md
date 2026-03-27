# Q4 Optimization Experiments (M3 Baseline + M4 Port + M1 Pro Port)

## Summary

This document tracks three phases:

1. Original M3 Max optimization pass (48GB), culminating at 4.4 tok/s.
2. M4 16GB port + optimization pass, culminating at 11.5 tok/s and 2.5s TTFT.
3. M1 Pro 16GB port + optimization pass, culminating at 11.91 tok/s and 1.71s TTFT at K=4.

## Results Comparison

| Phase | Hardware | K | Sustained tok/s | TTFT | Outcome |
|---|---|---:|---:|---:|---|
| Original baseline | M3 Max 48GB | 4 | 4.4 | ~5.6s | First production-quality release |
| M4 port | M4 16GB | 6 | **11.5** | **2.5s** | 2.6× faster on lower-cost hardware |
| M1 Pro port | M1 Pro 16GB | 4 | **11.91** | **1.71s** | Exceeds M4 reference on older silicon |

## M3 Max Track (Historical)

The original optimization series established:

- Stable 4-bit quality with tool calling.
- SSD streaming as the practical path for routed experts.
- Baseline pipeline behavior and first-wave kernel tuning.

Representative kept/discarded experiments from that phase remain relevant for context, including cache strategy and prefetch behavior under Apple unified memory.

## M1 Pro 16GB Port: What Changed

The M1 Pro pass started from tayoun's M4-optimized codebase and re-tuned for M1 Pro's
different hardware characteristics: slower SSD (~5.3 GB/s vs M4's ~17.5 GB/s), higher
memory bandwidth (200 GB/s vs M4's 120 GB/s), and more GPU cores (14 vs 10).

### 1. `tg256` Expert Matvec Kernels

tayoun's `tg128` was tuned for M4's 10-core GPU and underperformed on M1 Pro.
Switching to `tg256` better saturates M1 Pro's 200 GB/s memory bandwidth and
utilizes its 14 GPU cores more effectively.

- CMD1 improvement: −0.106 ms/layer
- CMD2 improvement: −0.082 ms/layer
- Net throughput gain at K=4: +0.98 tok/s (+9% over baseline-sweep)

**Why not tg512?** At `in_dim=2048`, tg512 gives each thread only 4 loop iterations
before the reduction phase dominates. Combined with higher synchronization cost and
reduced occupancy (2 threadgroups/core vs 4), tg512 adds overhead without additional
bandwidth gain — 200 GB/s is already saturated at tg256. tg256 is the sweet spot for
M1 Pro's specific core count and bandwidth profile.

### 2. K Tuning for M1 Pro

The M4 production default of K=6 is suboptimal on M1 Pro due to its slower SSD.
Each additional expert adds ~1.77 MB × 40 layers of SSD I/O per token.

- Recommended: `K=4` (11.91 tok/s, 1.71s TTFT) — best quality/speed balance
- Maximum throughput: `K=3` (12.77 tok/s, 1.66s TTFT) — quality=pass verified
- K=6 on M1 Pro: 9.66 tok/s — 16% slower than M4 reference due to SSD bottleneck

## Current M1 Pro Production Profile

- Model: Qwen3.5-35B-A3B-4bit
- Hardware: MacBook Pro M1 Pro, 16GB unified memory
- Routing: `K=4`
- Performance: 11.91 tok/s sustained, 1.71s TTFT (git `089cb24`)
- Stability: quality=pass across all benchmark runs

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
