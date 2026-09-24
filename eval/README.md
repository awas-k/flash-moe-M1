# Quality eval

A scored regression gate for output quality across `--k` values.

`bench.sh` reports `quality=pass` whenever generation produced ≥32 non-empty tokens
(`bench.sh`, the Python block at the end) — a liveness check, not a quality check. So every
claim that a lower K is quality-neutral has been unmeasured. This harness measures it.

## Why a fixed prompt set works here

Sampling in `infer.m` is greedy `cpu_argmax` (`infer.m:7085`) with no temperature, top-p or
seed — those request fields are parsed nowhere. **Output is deterministic**: a given
(K, prompt) always yields the same text. That makes 20 prompts a genuine regression gate
rather than a noisy sample, and it makes any diff between two K values attributable to K alone.

## Run it

```bash
# All four K values (3, 4, 6, 8) — ~20 min, one server per K
./eval/sweep.sh

# Or a subset
./eval/sweep.sh 4 8

# Re-print the report without re-running
python3 eval/compare.py
```

`sweep.sh` starts one server per K, because `--k` is fixed at launch and the accept loop is
serial. Results land in `eval/results/k<K>.json`, server logs beside them.

To score against an already-running server:

```bash
python3 eval/run_eval.py --port 8000 --k 4
```

The `--k` flag there only *labels* the output — it must match how the server was actually
started, since K cannot be set per request.

## The prompt set

20 prompts in `eval/prompts.jsonl`, each with an objectively checkable answer, across six
categories: `arithmetic`, `factual`, `format`, `code`, `multilingual`, `domain`.

Three are regressions observed in real K=4 output, kept as permanent guards:

| id | Guards against |
|---|---|
| `moe-02` | Claiming Switch Transformer activates 2 experts per token. It is **top-1** — that is the paper's whole contribution. |
| `ja-02` | Rendering the MoE router as 「ガバナー」("governor") instead of 「ルーター」. A gating→governor slip seen at K=4 but not K=8. |
| `moe-01` | The same governor/router confusion in English. |

Every prompt constrains its output format ("answer with only the number"), which keeps grading
robust without an LLM judge.

## Graders

A prompt passes only if **every** grader key it declares passes. Available keys:

| key | meaning |
|---|---|
| `int_eq` | first **or** last integer in the answer equals the value (tolerates "The answer is 391.") |
| `any_of` | at least one string present, case-insensitive |
| `none_of` | none of these present — used for the regression guards |
| `regex` | pattern matches (multiline) |
| `json_eq` | answer parses as JSON, with these keys and values; strips code fences |
| `csv_len` | exactly N comma-separated items, and none long enough to be smuggled prose |

`<think>…</think>` is stripped before grading, since reasoning streams inline as ordinary
content and is not part of the answer. An unterminated `<think>` means the cap truncated it
mid-reasoning, and everything after is discarded.

Adding a prompt means appending one JSONL line — no code change.

## Metrics needing no ground truth

Collected per prompt alongside pass/fail:

- `tok_s`, `ttft_s` — throughput, so the quality cost of a K can be weighed against its speed gain.
- `hit_cap` — generation stopped at `max_tokens`. Worth knowing because `finish_reason` is
  hardcoded `"stop"` (`infer.m:6419`) and never `"length"`, so the server never reports truncation.
- `replacement_chars` — count of U+FFFD, which detects the `sse_send_delta` UTF-8 splitting bug
  (`infer.m:6355`): a character split across byte-level tokens is corrupted in transit.
- `rep_rate` — fraction of duplicate 4-grams, catching degeneracy loops.

## Reading the report

`compare.py` prints pass rate overall and per category, the throughput/degeneracy table, and
two lists that matter more than the headline number:

- **Prompts that pass at the highest K but fail lower** — the actual cost of reducing K, with
  the failing answer quoted.
- **Prompts failing at the highest K too** — model limitations, not routing damage. Do not
  count these against a lower K.

It also reports exact-text agreement with the highest K present. Treat that as a divergence
signal, not a quality score: differing text can still be correct.

## Caveats

Isolation is per prompt — each is sent with no `session_id`, so every item restores the cached
system-prompt snapshot and items cannot contaminate each other. Multi-turn behaviour is
therefore **not** covered here; see `docs/web-chat.md`.

20 prompts is enough to catch the terminology and factual-precision failures that show up when
K is cut, and far too few to be a benchmark. Treat a pass-rate difference of one or two items
as noise in coverage, not a measured effect; look at *which* prompts moved.
