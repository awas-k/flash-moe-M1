# Claude Code Autoresearch Prompt — Flash-MoE M1 Pro Optimization

Copy and paste everything below this line into Claude Code CLI as your opening prompt.

---

Read CLAUDE.md first. That file contains the full project context, hardware specs, repo structure, and optimization rules you must follow.

## Your mission

You are optimizing flash-moe inference performance on an M1 Pro MacBook Pro (14-core GPU, 16 GB RAM). The current baseline is 9.16 tok/s at K=4. Your goal is to find configurations or code changes that improve sustained tok/s without breaking output quality.

## Phase 1: Establish a proper baseline (do this first)

Run the parameter sweep to measure K=3,4,5 with variance data:

```bash
./bench_matrix.sh --k-values "3 4 5" --runs 3 --tag baseline-sweep
```

Then analyze:

```bash
python analyze_results.py --tag baseline-sweep
```

Report the results. Identify which K value gives the best tok/s with quality=pass. This is our true baseline with variance.

## Phase 2: Profile the bottleneck

After the baseline sweep, investigate where time goes per token. Run inference with timing:

```bash
./metal_infer/infer \
  --model "$MODEL_DIR" \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 4 \
  --prompt "Explain quantum computing" \
  --tokens 50 \
  --timing
```

From the per-layer timing output, calculate:
1. Average GPU time per layer (CMD1 + CMD2 + CMD3)
2. Average SSD I/O time per layer
3. Average CPU routing time per layer
4. What percentage of total per-layer time is SSD I/O vs GPU vs CPU

This tells us exactly where to focus optimization effort.

## Phase 3: Investigate one optimization at a time

Pick the highest-priority unchecked item from the optimization checklist in CLAUDE.md. For each experiment:

1. Describe what you plan to change and why
2. Show me the specific code diff before applying it
3. Wait for my approval before modifying any files
4. After approval, make the change, rebuild (`cd metal_infer && make clean && make infer && cd ..`)
5. Run `./bench_matrix.sh --quick --tag <experiment-name>`
6. Run `python analyze_results.py --compare baseline-sweep <experiment-name>`
7. Report: did tok/s improve? By how much? Any quality regression?
8. If improvement > 2%, keep the change. If regression, revert immediately.
9. Check off the item in CLAUDE.md and add a note with the result.

## Important rules

- NEVER modify files in packed_experts/, out_35b/model_weights.bin, tokenizer.bin, or vocab.bin
- NEVER skip the benchmark step — every change gets measured
- ONE change at a time — never stack multiple changes before benchmarking
- Always show me the diff and wait for approval before editing infer.m or shaders.metal
- If a build fails, fix it before moving on — never leave the repo in a broken state
- If you are unsure whether a change is safe, ask me instead of guessing
- Log everything to results.tsv — this is our experiment journal

## Communication style

- Be concise. State what you plan to do, why, and expected impact in 2-3 sentences.
- Show code diffs as minimal context patches, not full file dumps.
- After each benchmark, give me a one-line verdict: "K=3 → 10.4 tok/s (+14%), quality=pass. KEEP." or "tg64 kernels → 8.8 tok/s (-4%), quality=pass. REVERT."
- At the end of each session, summarize: what we tried, what worked, current best tok/s.

Start with Phase 1 now.
