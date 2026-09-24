# Flash-MoE M1 Pro Optimization

## Project context

This is a fork of [tayoun/flash-moe](https://github.com/tayoun/flash-moe) (itself adapted from [danveloper/flash-moe](https://github.com/danveloper/flash-moe)).
It runs Qwen3.5-35B-A3B (35B param MoE, 256 experts, 40 layers) on Apple Silicon
via pure C/Objective-C + Metal shaders, streaming expert weights from SSD.

**Target hardware**: MacBook Pro M1 Pro, 10-core CPU, 14-core GPU, 16 GB unified memory.

**Current baseline** (git `8a217dd`, 256-token runs, warm page cache):
- K=4: **11.53 tok/s** sustained, **1.94s** TTFT — the recommended profile
- K=3: 12.60 tok/s, 1.75s TTFT — faster, but **quality-degraded**; see [Choosing K](#choosing-k)

> These are the best *reproducible* figures: `8a217dd` is the newest commit containing the
> tg256 change. The 11.91 / 12.77 numbers quoted previously came from an uncommitted tree and
> do not reproduce from any commit — see [M1 Pro–Specific Optimizations](#m1-prospecific-optimizations).
> Expect less in real use: sustained chat with a growing context runs ~9.5–10.5 tok/s, since
> these benchmarks are short generations from an empty context.

**Reference results**:
- M4 Mac mini 16 GB, K=6: 11.5 tok/s, 2.5s TTFT (tayoun)
- M3 Max MBP 48 GB, K=4: 4.4 tok/s, ~5.6s TTFT (danveloper)


## Results

All runs: 256 output tokens, warm page cache, no crashes.
tok/s and TTFT are per-run values (not averaged) unless noted.

> These are throughput figures only. `bench.sh` reports `quality=pass` whenever ≥32 non-empty
> tokens came back — a liveness check, not a quality measurement. Output quality is measured
> separately by [`eval/`](eval/README.md); see [Choosing K](#choosing-k).

| Machine | Model | Tag | K | tok/s | TTFT | Notes |
|---|---|---|---:|---:|---:|---|
| M3 Max MBP (48 GB) | Qwen3.5-35B-A3B-4bit | danveloper baseline | 4 | 4.4 | ~5.6s | Original project baseline |
| M4 Mac mini (16 GB) | Qwen3.5-35B-A3B-4bit | tayoun reference | 6 | 11.5 | 2.5s | tg128 kernel, tayoun fork |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline | 4 | 9.16 | 2.39s | Initial bring-up, bench.sh single run |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 3 | 11.77 | 1.88s | avg 3 runs, before kernel tuning |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 4 | 10.93 | 1.98s | avg 3 runs, before kernel tuning |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 5 | 10.14 | 2.19s | avg 3 runs, before kernel tuning |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | tg256-expert | 3 | 12.77 | 1.66s | avg 3 runs — uncommitted tree; also quality-degraded |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | tg256-expert | 4 | 11.91 | 1.71s | avg 3 runs — uncommitted tree, does not reproduce |
| **M1 Pro MBP (16 GB)** | **Qwen3.5-35B-A3B-4bit** | **baseline** | **4** | **11.53** | **1.94s** | **git `8a217dd`, avg 2 runs — recommended profile** |

> New `baseline` run (git `8a217dd`, 2 runs/K): K=3: 12.60 tok/s / 1.75s TTFT, K=4: 11.53 tok/s / 1.94s TTFT, K=5: 10.61 tok/s / 2.13s TTFT, K=6: 9.66 tok/s / 2.44s TTFT


## Hardware

| Machine | CPU | GPU cores | Unified Memory | SSD read | Role |
|---|---|---:|---:|---:|---|
| MacBook Pro M1 Pro | M1 Pro (10-core) | 14 | 16 GB | ~5.3 GB/s (measured) | **This fork — optimization target** |
| Mac mini M4 | M4 | 10 | 16 GB | not measured | tayoun reference machine |
| MacBook Pro M3 Max | M3 Max | 40 | 48 GB | — | danveloper original baseline |

> M1 Pro figure measured with `F_NOCACHE` sequential reads over the packed expert files:
> 5.14–5.34 GB/s single-stream. This README previously claimed the M4 reads at ~17.5 GB/s and
> that the M1 Pro SSD is therefore 3.3× slower. That figure was never measured here and is not
> achievable by an NVMe SSD — it exceeds the ~8 GB/s ceiling of PCIe 4.0 ×4 — so the ratio
> built on it has been removed.
>
> M1 Pro does have 1.7× the memory bandwidth of the base M4 (200 vs 120 GB/s) and 1.4× the GPU
> cores (14 vs 10).


## Architecture

- Qwen3.5-35B-A3B MoE inference implemented in C/Objective-C + Metal.
- Non-expert weights loaded once (`model_weights.bin`); expert weights stream from SSD at token time.
- Routing **K is runtime-configurable** (`--k`).
- Per-token SSD I/O at K=4: 4 experts × 1.77 MB × 40 layers ≈ **283 MB/token** from SSD.

### Where the time actually goes

Per-layer breakdown at K=4, from the runtime's own `-T` flag (avg of 3800 layers):

| Phase | ms/layer | share |
|---|---:|---:|
| `cmd1_wait` (GPU) | 0.927 | 45% |
| `expert_io` (SSD) | 0.641 | 31% |
| `cmd2_wait` (GPU) | 0.435 | 21% |
| everything else | 0.075 | 4% |

**GPU synchronisation accounts for ~65% of token time; expert I/O for ~31%.** Earlier revisions
of this README described the SSD as "the primary bottleneck" — measurement does not support
that. The engine issues 3 command buffers and 2 CPU↔GPU synchronisations per layer, i.e. 120
buffers and 80 round-trips per token, to move ~34 MB/layer that 200 GB/s should cover in
~0.17 ms against 1.36 ms observed. The remaining headroom is in launch and synchronisation
overhead, not in storage.

Reproduce with `-T`:

```bash
./metal_infer/infer --model models/Qwen3.5-35B-A3B-4bit \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin --k 4 -t 96 -T -P "your prompt"
```

### M1 Pro–Specific Optimizations

- **`tg256` expert matvec selection** (`beb9e47`): tayoun's `tg128` variant was tuned for M4's
  10-core GPU and underperforms on M1 Pro, so the selection heuristic that picks it was removed
  and the pre-existing 256-thread `matvec_v3` is used throughout.
  - CMD2 improvement: −0.082 ms/layer
  - CMD1 improvement: −0.106 ms/layer
  - Net gain at K=4: **~+0.6 tok/s (+5.5%)** — 10.93 (`089cb24`) → 11.53 (`8a217dd`)

  Two corrections to earlier revisions of this README, both worth knowing before trusting the
  numbers above:

  1. **No kernel was written.** `beb9e47` changes only 17 lines of `infer.m` and does not touch
     `shaders.metal`; it deletes the `tg128` selection branch so control falls through to a
     kernel that already existed. The `dequant_matvec_4bit_v3_tg128` pipeline is still compiled
     and is now dead code. "tg256 kernels" overstated the change.
  2. **11.91 tok/s is not reproducible from any commit.** The `tg256-expert` rows in
     `results.tsv` are stamped `089cb24` — *"license: credit all contributors"*, dated two days
     **before** tg256 existed — so they were measured on an uncommitted tree and labelled with
     the then-HEAD SHA. Every sweep after `beb9e47` does contain tg256 and lands K=4 at
     **11.11–11.58**, never 11.91. The +9% figure compared that unreproducible run against the
     pre-tg256 baseline; committed-vs-committed the gain is +5.5%, and even that is confounded
     with the other changes in `beb9e47`.

### Upstream M4 Optimizations (tayoun)

- `tg128` matvec kernels optimized for M4's 10-core GPU.
- Encoder coalescing to reduce launch/synchronization overhead in prefill/encode paths.
- Kernel fusion in critical hot paths to reduce memory traffic and CPU–GPU handoff.


## Quick Start

### 1. Set up Python tools

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy tokenizers
```

### 2. Point to your local Hugging Face snapshot

```bash
export MODEL_DIR="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/<snapshot_id>}"
```

### 3. Build model artifacts

```bash
python3 build_expert_index_35b.py --model-path "$MODEL_DIR" --out expert_index_35b.json
python3 repack_experts_35b.py --index expert_index_35b.json
python3 metal_infer/extract_weights_35b.py --model "$MODEL_DIR" --output metal_infer/out_35b
python3 metal_infer/export_tokenizer_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/tokenizer.bin
python3 metal_infer/export_vocab_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/vocab.bin
```

### 4. Build runtime

```bash
cd metal_infer
make infer chat
cd ..
```

### 5. Run server

```bash
./metal_infer/infer \
  --model "$MODEL_DIR" \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 4 \
  --serve 8000
```

> **M1 Pro recommendation**: `--k 4`. It matches the model's trained top-8 routing on the
> quality eval while running 1.44× faster. Do **not** use `--k 3` — see [Choosing K](#choosing-k).
> tayoun's upstream default `--k 6` is optimized for M4 and will be slower on M1 Pro (~9.66 tok/s).

### 6. Smoke test

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain mixture-of-experts in one paragraph."}],"max_tokens":128,"stream":true}'
```

### 7. Benchmark

```bash
# Single run
K=4 ./bench.sh

# Multi-config sweep (appends to results.tsv)
./bench_matrix.sh
```


## Choosing K

`--k` truncates the router's trained top-8 (`num_experts_per_tok: 8`) to the top K and
renormalises, so every K below 8 discards routing mass the model was trained to use. What that
costs is measured by [`eval/`](eval/README.md) over 26 scored prompts — sampling is greedy and
deterministic, so differences are attributable to K alone.

| K | eval pass | tok/s | cap hits | repetition | agreement w/ K=8 | |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 23/26 | 11.22 | 6 | 0.041 | 54% | **degraded — avoid** |
| 4 | 24/26 | 10.29 | 3 | 0.009 | 58% | **fastest safe choice** |
| 5 | 25/26 | 9.36 | 1 | 0.001 | 65% | **most margin for the price** |
| 6 | 25/26 | 8.36 | 1 | 0.000 | 65% | superseded by K=5 |
| 8 | 24/26 | 7.16 | 2 | 0.000 | — | trained routing, slowest |

**Use K=4 or K=5.** Both pass everything trained top-8 routing passes. K=4 is 10% faster;
K=5 has 3× fewer token-cap hits and 9× less repetition. Pick on whether you want throughput
or margin — the one-item pass difference between them is noise on 26 prompts.

**K=6 is strictly worse than K=5**: identical pass rate, identical agreement with K=8,
identical cap hits, and 12% slower. There is no configuration on this hardware where K=6 is
the right choice, upstream default or not.

**K=3 is degraded**, and the failure is verbosity rather than wrong facts: it stops stopping.
A one-word translation came back as 370 characters; a 2–3 sentence question came back as 1814
characters with the model's own drafting process left in. The earlier "use K=3 for maximum
throughput" advice predates this measurement and is withdrawn.

That failure mode has a gradient behind it, which is the useful part for tuning. Repetition
runs 0.041 → 0.009 → 0.001 → 0.000 and cap hits 6 → 3 → 1 → 1 across K=3, 4, 5, 6: verbosity
falls steadily as K rises, and K=3 is the point where it breaks the output. K=4 is safe, but
it sits closer to that edge than K=5 does.

Re-run it yourself with `./eval/sweep.sh`. Full analysis in
[`docs/optimization-experiments-q4.md`](docs/optimization-experiments-q4.md).


## Repo Notes

- Core runtime: `metal_infer/infer.m`, `metal_infer/shaders.metal`
- Chat client: `metal_infer/chat.m`
- Web chat UI: `web/chat.html` — see `docs/web-chat.md`
- Benchmark (single run): `bench.sh`
- Benchmark (sweep): `bench_matrix.sh`
- Quality eval: `eval/` — see `eval/README.md`
- Experiment log: `results.tsv`
- Experiment notes: `docs/optimization-experiments-q4.md`
- Technical paper: `paper/flash_moe.pdf`


## License

MIT — see [LICENSE](LICENSE).

---
Copyright (c) 2025 Daniel Woods  
Copyright (c) 2025 Sujit Baruwal (35B model adaptation)  
Copyright (c) 2026 Anthony Tayoun (M4 optimization, 2.6x performance improvement)  
Copyright (c) 2026 awas-k (M1 Pro adaptation, 11.53 tok/s on M1 Pro 16GB)
