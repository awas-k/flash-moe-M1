# Flash-MoE M1 Pro Optimization

## Project context

This is a fork of tayoun/flash-moe (itself adapted from danveloper/flash-moe).
It runs Qwen3.5-35B-A3B (35B param MoE, 256 experts, 40 layers) on Apple Silicon
via pure C/Objective-C + Metal shaders, streaming expert weights from SSD.

**Target hardware**: MacBook Pro M1 Pro, 10-core CPU, 14-core GPU, 16 GB unified memory.

**Current baseline**: 11.91 tok/s sustained, 1.83s TTFT at K=4. (K=3: 12.77 tok/s, 1.66s TTFT)

**Reference results**:
- M4 Mac mini 16 GB, K=6: 11.5 tok/s, 2.5s TTFT
- M3 Max MBP 48 GB, K=4: 4.4 tok/s, ~5.6s TTFT

## Repository structure

```
metal_infer/
  infer.m              # Main inference engine (~7000 lines, Objective-C)
  shaders.metal        # Metal GPU kernels (~1200 lines)
  chat.m               # Interactive chat TUI
  tokenizer.h          # C BPE tokenizer
  Makefile             # Build: make infer chat
  extract_weights_35b.py
  export_tokenizer_35b.py
  export_vocab_35b.py
  out_35b/             # model_weights.bin + model_weights.json

build_expert_index_35b.py   # Maps expert tensors to file offsets
repack_experts_35b.py       # Packs experts into per-layer binaries
bench.sh                    # Single-run benchmark (starts server, runs one prompt)
bench_matrix.sh             # Multi-config sweep benchmark (this fork)
results.tsv                 # Experiment log
```

## Key architecture facts

- Model: 40 layers, hidden_dim=2048, 256 experts/layer, expert_intermediate=512
- Layer pattern: 30 GatedDeltaNet (linear attention) + 10 full attention (3:1 ratio)
- Expert size: 1,769,472 bytes each (4-bit quantized)
- Total expert data on disk: ~17 GB
- Non-expert weights: ~1.38 GB (mmap'd)
- Per-token I/O at K=4: 4 experts × 1.77 MB × 40 layers = ~283 MB from SSD

## M1 Pro hardware constraints

- SSD sequential read: ~5.3 GB/s (vs M4's ~17.5 GB/s — 3.3× slower)
- Memory bandwidth: 200 GB/s (vs M4's 120 GB/s — 1.7× faster)
- GPU cores: 14 (vs M4's 10)
- Page cache available: ~10 GB (16 GB total - OS - app - Metal buffers)
- The SSD is the primary bottleneck, not the GPU

## Optimization goals

1. Maximize sustained tok/s on M1 Pro 16 GB
2. Minimize TTFT (time to first token)
3. Maintain output quality (quality=pass in bench.sh)
4. Do not increase peak memory beyond safe limits (~6 GB app + ~10 GB page cache)

## How to run experiments

### Quick single test
```bash
K=4 ./bench.sh
```

### Parameter sweep
```bash
./bench_matrix.sh
```
Results append to `results.tsv`. Each row is one experiment with config + metrics.

### Build after code changes
```bash
cd metal_infer && make clean && make infer chat && cd ..
```

### Key files to modify for optimization

- `metal_infer/shaders.metal` — GPU kernel threadgroup sizes, tiling, FMA patterns
- `metal_infer/infer.m` — pipeline structure, command buffer management, I/O dispatch
- `metal_infer/Makefile` — compiler flags

## Experiment tracking

All experiments MUST be logged to `results.tsv` via bench_matrix.sh or manually.
Format: TSV with columns defined in the header.

Before making any code change:
1. Record the current baseline with bench_matrix.sh
2. Make the change
3. Re-run bench_matrix.sh with the same configs
4. Compare results in results.tsv
5. If regression > 2%, revert

## Rules for Claude Code

- NEVER modify model weight files, tokenizer files, or packed_experts/
- NEVER change the API contract (request/response format of /v1/chat/completions)
- ALWAYS run bench.sh after modifying infer.m or shaders.metal to verify no regression
- ALWAYS preserve the existing K=4 baseline before testing other K values
- Keep Metal shaders compilable — test with `make infer` before claiming success
- When modifying shaders.metal, change ONE kernel at a time, benchmark, then proceed
- Commit messages: "exp: <description> — <tok/s result> tok/s (was <baseline>)"

## Optimization ideas to investigate (priority order)

### High priority — SSD pressure reduction
- [x] Benchmark K=3 vs K=4 vs K=5 (quality vs speed tradeoff) — K=3: 12.77 tok/s, K=4: 11.91, K=5: ~11 est. All quality=pass.
- [ ] Profile expert routing frequency — identify "hot" experts
- [x] Test small pinned expert cache (~500 MB) for top-N hot experts per layer — WORSE. 256MB cache: 8.35 tok/s (−21% vs 10.54 baseline). Trust the OS wins at 1.7:1 ratio too. malloc cache competes with page cache for same physical RAM. Flag --cache-mb kept but default=0.
- [ ] Measure actual page cache hit rate during sustained generation

### Medium priority — GPU kernel tuning for M1 Pro
- [x] Test original danveloper threadgroup sizes vs tayoun tg128 variants — tg256 wins +9% on M1 Pro (14-core). tg128 was M4-specific, hurts M1 Pro. APPLIED.
- [x] Test wider threadgroups to exploit M1 Pro's 200 GB/s bandwidth — confirmed: tg256 better. CMD2 −0.082ms/layer, CMD1 −0.106ms/layer.
- [ ] Profile Metal shader occupancy via Xcode GPU profiler
- [x] Benchmark with/without FMA dequant kernel on M1 Pro specifically — FMA (matvec_v3) is the exclusive active kernel on all hot paths. matvec_fast never fires (in_dim always ≤ 4096), v5/LUT is dead in serve mode. No accidental fallback.

### Low priority — pipeline experiments
- [ ] Test overlapping SSD pread with CMD1 GPU dispatch on M1 Pro
- [ ] Measure CMD1/CMD2/CMD3 timing breakdown with --timing flag
- [ ] Test F_NOCACHE vs page-cached reads at different generation lengths

### Speculative — quality-preserving compression
- [ ] 2-bit expert quantization (measure quality impact at 35B scale)
- [ ] Expert pruning — skip low-weight experts below threshold
