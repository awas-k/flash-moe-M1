# Flash-MoE: Miniforge Conda Setup for M1 Pro

## Complete environment setup using miniforge conda

### Step 1: Create the conda environment

```bash
# Create a dedicated environment with Python 3.11
# (3.11 is the sweet spot — well-tested, good ARM performance)
conda create -n flash-moe python=3.11 -y
conda activate flash-moe
```

### Step 2: Install Python dependencies

The repo needs only two pip packages for the prep scripts:

```bash
pip install numpy tokenizers
```

That's it for the Python side. The prep scripts (`build_expert_index_35b.py`,
`repack_experts_35b.py`, `extract_weights_35b.py`, `export_tokenizer_35b.py`,
`export_vocab_35b.py`) read safetensors files via raw binary I/O — they
don't need the `safetensors` Python library.

For downloading the model you also need:

```bash
pip install huggingface_hub hf_transfer
```

### Step 3: Make sure Xcode Command Line Tools are installed

The C/Metal runtime needs `clang` and the Metal compiler:

```bash
xcode-select --install    # skip if already installed
```

### Step 4: Clone and enter the repo

```bash
git clone https://github.com/tayoun/flash-moe.git
cd flash-moe
```

### Step 5: Download the 4-bit model

```bash
# Enable fast downloads via hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir ~/models/Qwen3.5-35B-A3B-4bit
```

This downloads ~18 GB. Using `--local-dir` gives you a clean path
instead of the nested `.cache/huggingface/hub/models--…/snapshots/…`
structure.

Then set:

```bash
export MODEL_DIR="$HOME/models/Qwen3.5-35B-A3B-4bit"

#my env
export MODEL_DIR="$HOME/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit
```

### Step 6: Build model artifacts (Python prep)

```bash
# Make sure conda env is active
conda activate flash-moe

# 1. Build expert index
python build_expert_index_35b.py --model-path "$MODEL_DIR" --out expert_index_35b.json

# 2. Repack experts into per-layer binary files (~17 GB output)
python repack_experts_35b.py --index expert_index_35b.json

# 3. Extract non-expert weights
python metal_infer/extract_weights_35b.py --model "$MODEL_DIR" --output metal_infer/out_35b

# 4. Export tokenizer
python metal_infer/export_tokenizer_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/tokenizer.bin

# 5. Export vocabulary
python metal_infer/export_vocab_35b.py "$MODEL_DIR/tokenizer.json" metal_infer/vocab.bin
```

### Step 7: Build C/Metal runtime

This step doesn't use Python/conda at all — it's pure clang + Metal:

```bash
cd metal_infer
make infer chat
cd ..
```

### Step 8: Run it

```bash
# Server mode (OpenAI-compatible API)
./metal_infer/infer \
  --model "$MODEL_DIR" \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 4 \
  --serve 8000

# Or interactive chat
./metal_infer/chat \
  --model "$MODEL_DIR" \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 4
```

Note: using `--k 4` instead of the default `--k 6` — better for your
16 GB M1 Pro (less SSD pressure, more room for page cache).

---

## Quick reference: disk space budget

| Item                          | Size     |
|-------------------------------|----------|
| Downloaded model (4-bit)      | ~18 GB   |
| packed_experts/ (repacked)    | ~17 GB   |
| model_weights.bin + manifest  | ~2-3 GB  |
| tokenizer + vocab             | ~10 MB   |
| **Total**                     | **~37-38 GB** |

Make sure you have ~40 GB free before starting.

After everything is built and verified, you can delete the original
downloaded safetensors to reclaim ~18 GB:

```bash
# Only after verifying inference works!
rm -rf ~/models/Qwen3.5-35B-A3B-4bit
```

---

## Troubleshooting

**`conda activate` doesn't work in zsh:**
```bash
conda init zsh
# Then restart your terminal
```

**`make` fails with "metal not found":**
Make sure Xcode CLT is installed and up to date:
```bash
sudo xcode-select --reset
xcode-select --install
```

**`huggingface-cli download` is slow:**
The `hf_transfer` speedup requires the env var:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**Out of memory during repack:**
The repack script uses `os.pread`/`os.pwrite` (streaming I/O), not
loading entire tensors into RAM. It should work fine on 16 GB. If
you still hit issues, repack a few layers at a time:
```bash
python repack_experts_35b.py --index expert_index_35b.json --layers 0-9
python repack_experts_35b.py --index expert_index_35b.json --layers 10-19
python repack_experts_35b.py --index expert_index_35b.json --layers 20-29
python repack_experts_35b.py --index expert_index_35b.json --layers 30-39
```
