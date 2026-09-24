# Browser chat testing

A single static page (`web/chat.html`) for chatting with the `--serve` endpoint in a browser,
in the spirit of llama-server's built-in web UI. No dependencies, no build step, works offline.

Use it to judge output quality by hand — particularly across `--k` values, which trade
routing fidelity for speed and are otherwise unmeasured (`bench.sh`'s `quality=pass` is a
liveness check: `tokens >= 32 and non-empty`, nothing more).


## Run it

Two terminals plus a browser.

```bash
# Terminal 1 — inference server
cd /Users/tb/Programs/qwen/flash-moe
./metal_infer/infer \
  --model models/Qwen3.5-35B-A3B-4bit \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin \
  --k 4 \
  --serve 8000
```

```bash
# Terminal 2 — static file server for the page
cd /Users/tb/Programs/qwen/flash-moe
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Then open **<http://127.0.0.1:8080/chat.html>**

Wait for Terminal 1 to print `[serve] Listening on http://0.0.0.0:8000` before sending
anything. Startup setup is ~120 ms, but the first request also pays prefill.


## Why two servers

`infer.m` serves no static content — there is no route for `/`, and no HTML anywhere in the
binary. The only three routes are:

| Method | Path | Returns |
|---|---|---|
| `POST` | `/v1/chat/completions` | SSE token stream |
| `GET` | `/v1/models` | `{"object":"list","data":[{"id":"qwen3.5-35b-a3b",...}]}` |
| `GET` | `/health` | `{"status":"ok","model":"qwen3.5-35b-a3b"}` |

So the page has to be served from somewhere else. That works without a proxy because the
server already sends `Access-Control-Allow-Origin: *` on every success path and answers
`OPTIONS` preflight with a 204 (`infer.m:6441`).


## Modes

The server does **not** read the `messages[]` array. `extract_last_content` (`infer.m:6237`)
scans the request body for the *last* `"content"` key and uses only that string. Conversation
history lives server-side in the KV cache and GatedDeltaNet state, keyed by a non-standard
`session_id` field.

**Continuation** (default) sends a stable `session_id`. When it matches the server's
`active_session_id`, the server takes the `is_continuation` path (`infer.m:6803`), prefixes
`<|im_end|>\n` to close the prior assistant turn, and resumes at `session_pos`.

Each turn still prefills its own new message, but never the history. That is cheaper than
resending the whole `messages[]` array, but **TTFT is not flat** — every new token still
attends across the whole KV cache, so it climbs as the conversation grows. Measured over a
three-turn chat (800-token replies):

| Turn | approx. context | TTFT K=4 | TTFT K=8 |
|---:|---:|---:|---:|
| 1 | ~20 tok | 1.85 s | 2.31 s |
| 2 | ~800 tok | 1.31 s | 1.79 s |
| 3 | ~1600 tok | 5.87 s | 7.30 s |

So expect a multi-second wait before the first token once a conversation passes roughly a
thousand tokens, at any K. Press New chat when it starts to drag.

**Stateless** omits `session_id`, so every turn restores the system-prompt snapshot and starts
fresh. Use it to isolate a bad answer from bad session state.

**New chat** issues a fresh `session_id`, which is what makes the server reset.

> Note: `metal_infer/chat.m` builds a full `messages` array but never sends `session_id`
> (`chat.m:314`), so it takes the `[NEW]` path every turn and its history is discarded. This
> page is the first client that actually exercises continuation.

There is no edit or regenerate button. With 30 GatedDeltaNet layers the recurrent state
cannot be rewound, so a turn cannot be un-said — New chat is the only reset.


## What the server ignores

Only `messages[].content`, `max_tokens` / `max_completion_tokens`, and `session_id` are parsed.
Everything else in the request is silently dropped: `temperature`, `top_p`, `top_k`, `stop`,
`n`, `model`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `tools`,
`response_format`. `stream` is ignored too — responses are **always** SSE.

Sampling is greedy `cpu_argmax` (`infer.m:7085`) with an anti-repetition rule that forces the
second-best token after 16 identical repeats (`infer.m:7000`). **Output is deterministic**, so
there is no sampling knob to tune and no need to re-roll a response.

`--k` is fixed at launch, not per-request. To compare quality across K, restart Terminal 1:

```bash
# Ctrl-C Terminal 1, then re-run with a different K (3, 4, 5, 6, or 8)
./metal_infer/infer --model models/Qwen3.5-35B-A3B-4bit \
  --weights metal_infer/out_35b/model_weights.bin \
  --manifest metal_infer/out_35b/model_weights.json \
  --vocab metal_infer/vocab.bin --k 8 --serve 8000
```

K=8 matches the model's trained `num_experts_per_tok`; lower K truncates the router's top-8 and
renormalizes, which is where the throughput gains come from. Ask the same question at two K
values and compare — output being deterministic makes that a clean A/B.


## Limits

**One request at a time.** The server is a single `accept` loop with no threads or fork
(`infer.m:6726`); each request runs prefill and generation inline before closing. A second tab
will sit in the listen backlog until the first finishes. The page disables Send and Check
server while streaming for this reason — and never polls in the background, which would
deadlock the server against itself.

**Context ceiling is ~8192 tokens**, not the 262144 in `config.json`. The CPU KV caches are
lazily allocated at `max_position_embeddings`, but the GPU KV mirror is fixed at 16.8 MB per
layer, which at `kv_heads=2 × head_dim=256 × 4B` is ~8200 positions. Nothing in the server
clamps this, so the page tracks a running estimate, warns at 7000, and blocks sending at 8192.
Press New chat.

**Reasoning streams inline.** Thinking tokens arrive as ordinary `content` — `in_think` only
gates session persistence (`infer.m:7034`), and there is no `reasoning_content` field. The page
splits `<think>…</think>` into a collapsed block.

**Emoji and some multi-byte characters arrive corrupted.** This is a server bug, not a page
bug, and no client can repair it. `sse_send_delta` (`infer.m:6355`) copies each token's bytes
straight into a JSON string with a byte-oriented escape loop that has no UTF-8 awareness. When
the tokenizer splits one character across several byte-level tokens — common for 4-byte emoji —
each chunk carries a *fragment* of the UTF-8 sequence, so each chunk's JSON string is
independently invalid and every standard decoder replaces it with U+FFFD. The halves can never
be rejoined, because the damage happens inside separate JSON strings before the client sees
them.

Reproduced by asking for five emoji: 👍, 🚀, 🔥 and 🎉 each arrived as several chunks
containing one U+FFFD apiece, while 😊 happened to be a single complete token and survived
intact. CJK text has been fine in testing — those tokens appear to hold whole characters — but
that is luck, not correctness. Fixing it means buffering incomplete UTF-8 sequences in
`sse_send_delta` and flushing only complete characters.


## Troubleshooting

| Symptom | Cause |
|---|---|
| `Server unreachable` | Terminal 1 not up yet, or a different port. Check its `[serve] Listening` line. |
| `Request failed` with no detail | The 400/500 replies omit CORS headers (`infer.m:6781`, `6798`, `6827`), so the browser cannot read the status. The real error is in Terminal 1, which logs every request. |
| Second tab hangs | Expected — serial server. Wait for the first generation to finish. |
| Model forgets context | Mode is Stateless, or `session_id` changed. Terminal 1 logs `[NEW]` vs `[CONTINUE]` per request. |
| `{"error":"not found"}` | Path mismatch. The request line is parsed with `sscanf("%15s %255s")` (`infer.m:6737`) and query strings are **not** stripped, so `/v1/models?x` 404s. There is no 405 — a wrong method also 404s. |
| `Stream truncated` | Server closed mid-generation without `[DONE]`. Check Terminal 1 for a crash or a failed write. |
| Message rejected before sending | The text contains a literal `"content"`, which would hijack the server's substring parse. |


## curl fallback

When the browser is not the thing under test:

```bash
# One turn (always streams, regardless of "stream")
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2?"}],"max_tokens":64,"session_id":"t1"}'

# Follow-up on the same session — reuse the session_id to keep context
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"And times three?"}],"max_tokens":64,"session_id":"t1"}'

curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```


## Security

`--serve` binds `INADDR_ANY` — all interfaces, not just localhost (`infer.m:6577`). There is no
`--host` flag and **no authentication of any kind**, so while the server is running, anyone on
your network can send it prompts. Binding `127.0.0.1` for the static file server does not
change that; it only limits who can fetch the page.

Run it only on networks you trust, and stop Terminal 1 when you are done. Adding a
`--host 127.0.0.1` option to `infer.m` would fix this properly.


## Third-party UIs

Open WebUI, LibreChat and similar resend the full `messages[]` array on every turn. This
server discards it and reads only the last `"content"`, and they do not send `session_id` — so
they appear to work while having no memory at all. They are not usable here without either
patching them to send a stable `session_id`, or changing `infer.m` to honor `messages[]`.
