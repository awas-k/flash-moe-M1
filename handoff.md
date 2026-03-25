## Handoff: flash-moe M1 Pro — Session Continuity Fix

### Context
This is a fork of flash-moe running Qwen3.5-35B-A3B on M1 Pro 16GB via custom Metal inference (`metal_infer/infer.m`). See `CLAUDE.md` for full project context and rules.

### What was just fixed
1. **Multi-turn conversation** — `extract_last_content()` was replaced with `build_qwen_prompt()` which builds a full Qwen3 chat-formatted prompt from the entire messages array. This fixed context carry-over.
2. **`/no_think` system prompt** — injected by default in `build_qwen_prompt()` to suppress extended thinking for normal chat.

### Current performance after fixes
- 11.28 tok/s, 2.52s TTFT at K=4 (was 11.91 / 1.83s before multi-turn fix)

### The problem to fix now
**KV cache is never reused between turns.** Every request shows `session=(none) [NEW]` in server logs. The server re-prefills the entire conversation history from scratch on every turn, causing prefill time to grow with every message:

```
chatcmpl-3: 424 tokens → 36s prefill
chatcmpl-4: 513 tokens → 46s prefill  
chatcmpl-7: 1001 tokens → (blows up)
```

The infrastructure exists (`active_session_id`, `is_continuation`, `session_pos`) but session matching never triggers because Open WebUI doesn't send whatever session identifier `infer.m` is expecting.

### The fix to implement
**Implicit session continuity** — don't rely on the client sending a session ID. Instead, detect continuation server-side by checking if the new prompt is a prefix-extension of the cached prompt:

```c
// Pseudocode logic
if (new_prompt starts with cached_prompt) {
    // continuation — only prefill the new tail tokens
    start_pos = cached_token_count;
} else {
    // new conversation — full prefill from pos 0
    start_pos = 0;
}
```

This works because Open WebUI always sends the full conversation history, so each new turn's prompt is always the previous prompt + new messages appended.

### How to approach the code
Don't read all 365k of `infer.m` at once. Use targeted greps:

```bash
# Find session handling code
grep -n "is_continuation\|active_session\|session_pos\|session_id" metal_infer/infer.m

# Find where prefill starts
grep -n "prefill\|start_pos\|kv_pos\|cache_pos" metal_infer/infer.m

# Find where the prompt token array is built
grep -n "prompt_tokens\|n_prompt\|tokenize" metal_infer/infer.m

# Find build_qwen_prompt usage
grep -n "build_qwen_prompt\|extract_last_content" metal_infer/infer.m
```

Read only the relevant sections (~50-100 lines around each hit) rather than the full file.

### Rules (from CLAUDE.md)
- Run `bench.sh` after any change to `infer.m` to verify no regression
- Baseline to beat: **11.28 tok/s, 2.52s TTFT**
- Target: same tok/s, TTFT back toward ~1.83s for short continuations
- NEVER modify weight files, tokenizer files, or packed_experts/
- NEVER change the `/v1/chat/completions` API contract
- Commit message format: `exp: <description> — <tok/s> tok/s (was 11.28)`

### Success criteria
Server logs should show `[CONTINUATION]` (or equivalent) on turns 2+ of a conversation, with prefill time staying low (~2-3s) regardless of conversation length.