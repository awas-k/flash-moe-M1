#!/usr/bin/env python3
"""
flash_moe_api_samples.py — Sample API calls for the Flash-MoE server.

The server exposes an OpenAI-compatible streaming API at http://127.0.0.1:8000.
Note: this server always returns SSE (streaming) format, so all requests use stream=true.

Usage:
    python flash_moe_api_samples.py
    python flash_moe_api_samples.py --host 127.0.0.1 --port 8000
    python flash_moe_api_samples.py --test 4   # run only test 4
"""

import json
import argparse
import urllib.request
import urllib.error
import sys
import time
import re


BASE = "http://127.0.0.1:8000"


def stream_chat(messages, max_tokens=128):
    """Send a streaming chat request and return the full assembled response text."""
    url = f"{BASE}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)

    full_text = ""
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            full_text += token
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return full_text


def stream_chat_print(messages, max_tokens=256):
    """Send a streaming chat request, print tokens live, return the full text."""
    url = f"{BASE}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)

    full_text = ""
    token_count = 0
    t0 = time.time()

    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        body = line[6:]
        if body == "[DONE]":
            break
        try:
            chunk = json.loads(body)
            token = chunk["choices"][0].get("delta", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token
                token_count += 1
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    elapsed = time.time() - t0
    tps = token_count / elapsed if elapsed > 0 else 0
    print(f"\n  [{token_count} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]")
    return full_text


def api_get(endpoint):
    """GET from the API."""
    url = f"{BASE}{endpoint}"
    resp = urllib.request.urlopen(url)
    return json.loads(resp.read().decode("utf-8"))


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_health():
    print("=" * 60)
    print("1. Health Check")
    print("=" * 60)
    result = api_get("/health")
    print(json.dumps(result, indent=2))
    print()


def test_models():
    print("=" * 60)
    print("2. List Models")
    print("=" * 60)
    result = api_get("/v1/models")
    print(json.dumps(result, indent=2))
    print()


def test_simple_chat():
    print("=" * 60)
    print("3. Simple Chat")
    print("=" * 60)
    print("Q: What is the capital of Japan?")
    print("A: ", end="", flush=True)
    stream_chat_print(
        [{"role": "user", "content": "What is the capital of Japan? Answer in one sentence."}],
        max_tokens=64,
    )
    print()


def test_streaming():
    print("=" * 60)
    print("4. Streaming — Longer Response")
    print("=" * 60)
    print("Q: Explain mixture-of-experts in 3 sentences.")
    print("A: ", end="", flush=True)
    stream_chat_print(
        [{"role": "user", "content": "Explain what a mixture-of-experts model is in 3 sentences."}],
        max_tokens=256,
    )
    print()


def test_multi_turn():
    print("=" * 60)
    print("5. Multi-turn Conversation")
    print("=" * 60)

    messages = [{"role": "user", "content": "What is 25 * 37? Just give the number."}]

    print("User: What is 25 * 37?")
    print("Assistant: ", end="", flush=True)
    reply1 = stream_chat_print(messages, max_tokens=64)

    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "Now divide that result by 5. Just give the number."})

    print("User: Now divide that result by 5.")
    print("Assistant: ", end="", flush=True)
    stream_chat_print(messages, max_tokens=64)
    print()


def test_system_prompt():
    print("=" * 60)
    print("6. System Prompt — Poet Mode")
    print("=" * 60)
    print("(System: You are a poet who responds in haiku.)")
    print("Q: Describe a laptop running AI locally.")
    print("A: ", end="", flush=True)
    stream_chat_print(
        [
            {"role": "system", "content": "You are a poet who only responds in haiku (5-7-5 syllable format). No thinking, just the haiku."},
            {"role": "user", "content": "Describe a laptop running AI locally."},
        ],
        max_tokens=64,
    )
    print()


def test_code():
    print("=" * 60)
    print("7. Code Generation")
    print("=" * 60)
    print("Q: Write a Python is_prime function.")
    print("A: ", end="", flush=True)
    stream_chat_print(
        [{"role": "user", "content": "Write a short Python function that checks if a number is prime. No explanation, just the code."}],
        max_tokens=256,
    )
    print()


def test_japanese():
    print("=" * 60)
    print("8. Japanese Language")
    print("=" * 60)
    print("Q: 日本の四季について一文で説明してください。")
    print("A: ", end="", flush=True)
    stream_chat_print(
        [{"role": "user", "content": "日本の四季について一文で説明してください。"}],
        max_tokens=128,
    )
    print()


def print_curl_examples():
    print("=" * 60)
    print("curl Examples")
    print("=" * 60)
    print("""
# Health check
curl http://127.0.0.1:8000/health

# Streaming chat
curl -N http://127.0.0.1:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"stream":true}'

# With system prompt
curl -N http://127.0.0.1:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages":[
      {"role":"system","content":"You are a helpful coding assistant."},
      {"role":"user","content":"Write fizzbuzz in Python."}
    ],"max_tokens":256,"stream":true}'
""")


def main():
    global BASE
    parser = argparse.ArgumentParser(description="Flash-MoE API sample client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test", type=int, default=None, help="Run only test N (1-8)")
    args = parser.parse_args()
    BASE = f"http://{args.host}:{args.port}"

    tests = [
        test_health,
        test_models,
        test_simple_chat,
        test_streaming,
        test_multi_turn,
        test_system_prompt,
        test_code,
        test_japanese,
    ]

    print(f"\nFlash-MoE API Samples — {BASE}\n")

    try:
        if args.test:
            if 1 <= args.test <= len(tests):
                tests[args.test - 1]()
            else:
                print(f"Test {args.test} not found. Valid: 1-{len(tests)}")
        else:
            for t in tests:
                t()
            print_curl_examples()
    except urllib.error.URLError as e:
        print(f"\nERROR: Could not connect to {BASE} — {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
