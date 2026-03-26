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




## Results

updated results table should be:

| Machine | Model | K | tok/s | TTFT | Notes |
|---|---|---:|---:|---:|---|
| M3 Max MBP (48 GB) | Qwen3.5-35B-A3B-4bit | 4 | 4.4 | ~5.6s | Original baseline |
| M1 Pro MBP (16 GB) | Qwen3.5-35B-A3B-4bit | 4 | 9.16 | 2.39s | Before optimization |
| M4 Mac mini (16 GB) | Qwen3.5-35B-A3B-4bit | 6 | 11.5 | 2.5s | tayoun reference |
| **M1 Pro MBP (16 GB)** | **Qwen3.5-35B-A3B-4bit** | **4** | **11.91** | **1.71s** | **tg256 kernel, quality-verified** |


## Hardware

| Machine | CPU/GPU | Unified Memory | Role |
|---|---|---:|---|
| MacBook Pro (M3 Max) | M3 Max | 48GB | Original bring-up + baseline optimization |
| Mac mini (M4) | M4 | 16GB | Current optimized runtime target |

## Architecture

- Qwen3.5-35B-A3B MoE inference implemented in C/Objective-C + Metal.
- Non-expert weights are loaded once (`model_weights.bin`), expert weights stream from SSD at token time.
- Routing **K is runtime-configurable** (`--k`).
  - Original M3 tuning focused on `K=4`.
  - Current M4 production setup uses `K=6`.
- Pipeline remains SSD-aware and unified-memory-aware: optimize total token latency, not isolated kernel microbenchmarks.

### M4-Specific Optimizations

- `tg128` matvec kernels for better threadgroup utilization.
- Encoder coalescing to reduce launch/synchronization overhead in prefill/encode paths.
- Kernel fusion in critical hot paths to cut memory traffic and CPU-GPU handoff overhead.

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
  --k 6 \
  --serve 8000
```

### 6. Smoke test

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain mixture-of-experts in one paragraph."}],"max_tokens":128,"stream":true}'
```

## Repo Notes

- Core runtime: `metal_infer/infer.m`, `metal_infer/shaders.metal`
- Chat client: `metal_infer/chat.m`
- Benchmark helper: `bench.sh`
- Experiment notes: `docs/optimization-experiments-q4.md`
- Technical paper: `paper/flash_moe.pdf`

## License

MIT — see [LICENSE](LICENSE).
