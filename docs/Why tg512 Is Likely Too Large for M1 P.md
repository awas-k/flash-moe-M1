# Why tg512 Is Likely Too Large for M1 Pro

## Short Answer

**tg512 will probably hurt performance on M1 Pro.** The tg256 sweet spot balances memory bandwidth saturation against synchronization overhead. Going larger tips the balance the wrong way.

## How Apple GPU Execution Works

### SIMD-groups: The Fundamental Unit

Apple GPUs execute threads in groups of 32, called **SIMD-groups** (equivalent to NVIDIA warps).

```
tg128 =  4 SIMD-groups per threadgroup
tg256 =  8 SIMD-groups per threadgroup
tg512 = 16 SIMD-groups per threadgroup
```

### Per-Core Capacity (M1 Pro, Estimated)

Each of the 14 GPU cores has finite resources:

```
Max concurrent threads:     ~768–1024 per core
Max SIMD-groups:            ~24–32 per core
Threadgroup memory:         32 KB per core
Register file:              Fixed size, shared across all threads
```

## Occupancy Analysis

**Occupancy** = how many threadgroups a core can run simultaneously.

```
tg256:  1024 ÷ 256 = 4 threadgroups per core
        → 4 groups competing for scheduling → good latency hiding

tg512:  1024 ÷ 512 = 2 threadgroups per core
        → only 2 groups → less flexibility for the scheduler
```

Although both yield ~1024 in-flight threads, **fewer, larger groups give the scheduler less room to hide stalls**.

## Four Reasons tg512 Hurts

### Reason 1: Not Enough Work Per Thread

The hot kernel is `matvec_v3` — a matrix-vector multiply with 4-bit dequantization:

```metal
for (int i = tid; i < in_dim; i += THREADGROUP_SIZE) {
    partial_sum += dequantize(weight[i]) * input[i];
}
```

With `in_dim = 2048` (the model's hidden dimension):

```
tg256: each thread processes 2048 / 256 =  8 elements
tg512: each thread processes 2048 / 512 =  4 elements
                                           ↑ only 4 iterations!
```

At just 4 loop iterations, the **reduction phase dominates** the total work. The thread does almost nothing before hitting the barrier.

### Reason 2: Barrier Synchronization Costs Scale

The reduction requires all threads in a group to synchronize:

```
tg256: log2(256) = 8 reduction steps, synchronizing 256 threads
tg512: log2(512) = 9 reduction steps, synchronizing 512 threads
```

More importantly, the probability that **one slow thread stalls everyone** grows with group size:

```
tg256: ████████░░████████        (short sync wait)
tg512: ████████░░░░░░████████    (longer sync wait)
```

### Reason 3: Register Pressure

Each thread needs its own registers for local variables. The register file per core is fixed.

```
tg256 × 4 groups = 1024 threads sharing the register file
tg512 × 2 groups = 1024 threads sharing the register file
```

The thread count is similar, but the **compiler must guarantee registers for 512 threads in one group atomically**. This can force register spilling to memory, which is very expensive.

### Reason 4: Bandwidth Is Already Saturated

This is the most important point:

```
M1 Pro memory bandwidth: 200 GB/s

tg256 with 14 cores:
  14 cores × 4 groups × 256 threads = 14,336 in-flight threads
  → Enough memory requests to saturate 200 GB/s ✓

tg512 with 14 cores:
  14 cores × 2 groups × 512 threads = 14,336 in-flight threads
  → Same thread count, but with worse scheduling flexibility
  → No additional bandwidth gain, only more overhead
```

**You can't read faster than 200 GB/s no matter how many threads you add.** Beyond saturation, more threads only add overhead.

## The Sweet Spot Visualized

```
Perf ↑
     |
     |              ★ tg256 (sweet spot)
     |             ╱ ╲
     |            ╱   ╲
     |           ╱     ╲ tg512
     |          ╱       ╲
     |    tg128╱         ╲
     |        ╱           ╲ tg1024
     |       ╱
     +──────────────────────────→ threadgroup size
      32  64  128  256  512  1024

      ←─ bandwidth ─→← optimal →←── overhead dominates ──→
         starved
```

## Summary Table

| Factor | tg128 | tg256 | tg512 |
|---|---|---|---|
| Bandwidth utilization | Insufficient | **Saturated** | Saturated (no gain) |
| Loop iterations (in_dim=2048) | 16 | 8 | 4 (too few) |
| Reduction steps | 7 | 8 | 9 |
| Sync overhead | Low | Moderate | **High** |
| Groups per core | 6–8 | 4 | 2 (less flexible) |
| Register pressure | Low | Moderate | **High** |
| M1 Pro measured result | Baseline | **+9%** | Untested (likely worse) |

## Caveat

This analysis is based on estimated M1 Pro GPU specs (Apple doesn't publish internal details). **Empirical testing is the final word.** If you want to verify, create a tg512 variant in `shaders.metal` and run `bench.sh` — one experiment will confirm or refute the theory.

## Conclusion

**tg256 is likely the optimal threadgroup size for M1 Pro** because:

1. **Bandwidth is already saturated** at tg256 — more threads don't help
2. **in_dim=2048 gives only 4 iterations per thread** at tg512 — not enough useful work
3. **Synchronization cost grows** with group size while per-thread work shrinks
4. **Occupancy drops** from 4 to 2 groups per core, reducing scheduler flexibility

The +9% gain from tg128→tg256 came from better bandwidth utilization. Going to tg512 would add overhead without adding bandwidth — the wrong side of the curve.