# Flash-MoE M1 Pro Optimization

## Project context

This is a fork of [tayoun/flash-moe](https://github.com/tayoun/flash-moe) (itself adapted from [danveloper/flash-moe](https://github.com/danveloper/flash-moe)).
It runs Qwen3.5-35B-A3B (35B param MoE, 256 experts, 40 layers) on Apple Silicon
via pure C/Objective-C + Metal shaders, streaming expert weights from SSD.

**Target hardware**: MacBook Pro M1 Pro, 10-core CPU, 14-core GPU, 16 GB unified memory.

**Current baseline** (tg256 kernel, git `089cb24`):
- K=4: **11.91 tok/s** sustained, **1.71s** TTFT
- K=3: **12.77 tok/s** sustained, **1.66s** TTFT

**Reference results**:
- M4 Mac mini 16 GB, K=6: 11.5 tok/s, 2.5s TTFT (tayoun)
- M3 Max MBP 48 GB, K=4: 4.4 tok/s, ~5.6s TTFT (danveloper)


## Results

All runs: 256 output tokens, warm page cache, quality=pass.
tok/s and TTFT are per-run values (not averaged) unless noted.

| Machine | Model | Tag | K | tok/s | TTFT | Notes |
|---|---|---|---:|---:|---:|---|
| M3 Max MBP (48 GB) | Qwen3.5-35B-A3B-4bit | danveloper baseline | 4 | 4.4 | ~5.6s | Original project baseline |
| M4 Mac mini (16 GB) | Qwen3.5-35B-A3B-4bit | tayoun reference | 6 | 11.5 | 2.5s | tg128 kernel, tayoun fork |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline | 4 | 9.16 | 2.39s | Initial bring-up, bench.sh single run |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 3 | 11.77 | 1.88s | avg 3 runs, before kernel tuning |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 4 | 10.93 | 1.98s | avg 3 runs, before kernel tuning |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | baseline-sweep | 5 | 10.14 | 2.19s | avg 3 runs, before kernel tuning |
| **M1 Pro MBP (16 GB)** | **Qwen3.5-35B-A3B-4bit** | **tg256-expert** | **3** | **12.77** | **1.66s** | **avg 3 runs, tg256 kernel** |
| **M1 Pro MBP (16 GB)** | **Qwen3.5-35B-A3B-4bit** | **tg256-expert** | **4** | **11.91** | **1.71s** | **avg 3 runs, tg256 kernel** |

> New `baseline` run (git `8a217dd`, 2 runs/K): K=3: 12.60 tok/s / 1.75s TTFT, K=4: 11.53 tok/s / 1.94s TTFT, K=5: 10.61 tok/s / 2.13s TTFT, K=6: 9.66 tok/s / 2.44s TTFT


## Hardware

| Machine | CPU | GPU cores | Unified Memory | SSD read | Role |
|---|---|---:|---:|---:|---|
| MacBook Pro M1 Pro | M1 Pro (10-core) | 14 | 16 GB | ~5.3 GB/s | **This fork — optimization target** |
| Mac mini M4 | M4 | 10 | 16 GB | ~17.5 GB/s | tayoun reference machine |
| MacBook Pro M3 Max | M3 Max | 40 | 48 GB | — | danveloper original baseline |

> The M1 Pro SSD is ~3.3× slower than the M4, making SSD throughput the primary bottleneck.
> However, M1 Pro has 1.7× more memory bandwidth (200 GB/s vs 120 GB/s) and 1.4× more GPU cores (14 vs 10),
> which tg256 kernels exploit effectively.


## Architecture

- Qwen3.5-35B-A3B MoE inference implemented in C/Objective-C + Metal.
- Non-expert weights loaded once (`model_weights.bin`); expert weights stream from SSD at token time.
- Routing **K is runtime-configurable** (`--k`).
- Per-token SSD I/O at K=4: 4 experts × 1.77 MB × 40 layers ≈ **283 MB/token** from SSD.

### M1 Pro–Specific Optimizations

- **`tg256` expert matvec kernels**: Threadgroup size 256 better utilizes M1 Pro's 14-core GPU and
  200 GB/s memory bandwidth. tayoun's `tg128` was tuned for M4 and underperforms on M1 Pro.
  - CMD2 improvement: −0.082 ms/layer
  - CMD1 improvement: −0.106 ms/layer
  - Net gain at K=4: +0.98 tok/s vs baseline-sweep (+9%)

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

> **M1 Pro recommendation**: `--k 4` (11.91 tok/s, quality=pass). Use `--k 3` for maximum throughput (12.77 tok/s).
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


## Repo Notes

- Core runtime: `metal_infer/infer.m`, `metal_infer/shaders.metal`
- Chat client: `metal_infer/chat.m`
- Benchmark (single run): `bench.sh`
- Benchmark (sweep): `bench_matrix.sh`
- Experiment log: `results.tsv`
- Experiment notes: `docs/optimization-experiments-q4.md`
- Technical paper: `paper/flash_moe.pdf`


## License

MIT — see [LICENSE](LICENSE).
