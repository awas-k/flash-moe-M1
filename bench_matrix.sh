#!/usr/bin/env bash
# bench_matrix.sh — parameter sweep benchmark for flash-moe M1 Pro optimization
#
# Sweeps K values, measures tok/s + TTFT + quality, logs everything to results.tsv.
# Designed for autoresearch: run after code changes to detect improvements/regressions.
#
# Usage:
#   ./bench_matrix.sh                          # sweep K=3,4,5,6 with defaults
#   ./bench_matrix.sh --k-values "3 4 5"       # custom K values
#   ./bench_matrix.sh --runs 3                  # 3 runs per config (for variance)
#   ./bench_matrix.sh --tag "tg64-kernels"      # tag this experiment batch
#   ./bench_matrix.sh --tokens 128              # shorter generation
#   ./bench_matrix.sh --cold                    # purge page cache between runs (needs sudo)
#   ./bench_matrix.sh --quick                   # single run, K=4 only (CI smoke test)
# Environment:
#   CACHE_MB=256 ./bench_matrix.sh              # enable malloc expert cache (MB)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_FILE="${REPO_DIR}/results.tsv"
INFER="${REPO_DIR}/metal_infer/infer"
WEIGHTS="${WEIGHTS:-${REPO_DIR}/metal_infer/out_35b/model_weights.bin}"
MANIFEST="${MANIFEST:-${REPO_DIR}/metal_infer/out_35b/model_weights.json}"
VOCAB="${VOCAB:-${REPO_DIR}/metal_infer/vocab.bin}"
MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
PORT="${PORT:-8100}"

# Defaults
K_VALUES="3 4 5 6"
RUNS_PER_K=2
MAX_TOKENS=256
TAG="baseline"
COLD_CACHE=false
WARMUP_TOKENS=64
PROMPTS_FILE=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k-values)   K_VALUES="$2"; shift 2 ;;
        --runs)       RUNS_PER_K="$2"; shift 2 ;;
        --tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --tag)        TAG="$2"; shift 2 ;;
        --cold)       COLD_CACHE=true; shift ;;
        --quick)      K_VALUES="4"; RUNS_PER_K=1; MAX_TOKENS=128; TAG="quick"; shift ;;
        --warmup)     WARMUP_TOKENS="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate
if [[ -z "${MODEL_DIR}" ]]; then
    echo "[bench_matrix] ERROR: MODEL_DIR not set"
    exit 1
fi
if [[ ! -x "${INFER}" ]]; then
    echo "[bench_matrix] ERROR: infer binary not found. Run: cd metal_infer && make infer"
    exit 1
fi
for f in "${WEIGHTS}" "${MANIFEST}" "${VOCAB}"; do
    [[ -f "${f}" ]] || { echo "[bench_matrix] ERROR: missing ${f}"; exit 1; }
done

# Collect system info
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Total Number of Cores" | awk -F: '{print $2}' | tr -d ' ' || echo "?")
MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
GIT_SHA=$(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo "none")

echo "============================================================"
echo " Flash-MoE Benchmark Matrix"
echo "============================================================"
echo "  Chip:       ${CHIP}"
echo "  GPU cores:  ${GPU_CORES}"
echo "  Memory:     ${MEM_GB} GB"
echo "  macOS:      ${OS_VER}"
echo "  Git:        ${GIT_SHA}"
echo "  Tag:        ${TAG}"
echo "  K values:   ${K_VALUES}"
echo "  Runs/K:     ${RUNS_PER_K}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Cold cache: ${COLD_CACHE}"
echo "============================================================"
echo ""

# Create results file with header if it doesn't exist
if [[ ! -f "${RESULTS_FILE}" ]]; then
    printf "timestamp\ttag\tgit_sha\tchip\tgpu_cores\tmem_gb\tmacos\tK\tmax_tokens\trun\ttok_s\tttft_s\ttokens_out\tquality\tcold\tnotes\n" > "${RESULTS_FILE}"
    echo "[bench_matrix] Created ${RESULTS_FILE}"
fi

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_server() {
    local k_val="$1"
    cleanup

    local cache_flag=""
    if [[ "${CACHE_MB:-0}" -gt 0 ]]; then
        cache_flag="--cache-mb ${CACHE_MB}"
    fi

    # shellcheck disable=SC2086  # intentional word split for optional cache flag
    "${INFER}" \
        --model "${MODEL_DIR}" \
        --weights "${WEIGHTS}" \
        --manifest "${MANIFEST}" \
        --vocab "${VOCAB}" \
        --k "${k_val}" \
        ${cache_flag} \
        --serve "${PORT}" >/dev/null 2>&1 &
    SERVER_PID=$!

    # Wait for server ready
    for _ in $(seq 1 40); do
        if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[bench_matrix] ERROR: server crashed during startup (K=${k_val})"
            return 1
        fi
        sleep 0.5
    done
    echo "[bench_matrix] ERROR: server timeout (K=${k_val})"
    return 1
}

run_single_bench() {
    local max_tok="$1"
    local warmup="$2"

    python3 - "${PORT}" "${max_tok}" "${warmup}" <<'PYEOF'
import json, sys, time, urllib.request

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
warmup_tokens = int(sys.argv[3])
url = f"http://127.0.0.1:{port}/v1/chat/completions"

# Warmup run (short, results discarded)
if warmup_tokens > 0:
    warmup_payload = {
        "messages": [{"role": "user", "content": "Hello."}],
        "max_tokens": warmup_tokens,
        "stream": True,
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(warmup_payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                pass
    except Exception:
        pass
    time.sleep(0.5)

# Actual benchmark
payload = {
    "messages": [{"role": "user", "content": "Explain why mixture-of-experts improves compute efficiency."}],
    "max_tokens": max_tokens,
    "stream": True,
}
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})

start = time.time()
first_token_at = None
tokens = 0
text_parts = []

try:
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
            content = delta.get("content")
            if not content:
                continue
            if first_token_at is None:
                first_token_at = time.time()
            tokens += 1
            text_parts.append(content)
except Exception as e:
    print(f"0.00\t0.00\t0\tfail\t{e}", flush=True)
    sys.exit(0)

end = time.time()
ttft = (first_token_at - start) if first_token_at else 0.0
decode_s = (end - (first_token_at or end))
tok_s = (tokens / decode_s) if decode_s > 0 else 0.0

full_text = "".join(text_parts).strip()
quality = "pass" if tokens >= 32 and len(full_text) > 20 else "warn"

print(f"{tok_s:.2f}\t{ttft:.2f}\t{tokens}\t{quality}", flush=True)
PYEOF
}

purge_cache() {
    if [[ "${COLD_CACHE}" == "true" ]]; then
        echo "[bench_matrix] Purging page cache (requires sudo)..."
        sudo purge 2>/dev/null || echo "[bench_matrix] WARNING: purge failed (not root?)"
        sleep 2
    fi
}

# Main sweep
TOTAL_RUNS=0
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

for K in ${K_VALUES}; do
    echo "--- K=${K} ---"

    if ! start_server "${K}"; then
        # Log crash
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${TIMESTAMP}" "${TAG}" "${GIT_SHA}" "${CHIP}" "${GPU_CORES}" "${MEM_GB}" \
            "${OS_VER}" "${K}" "${MAX_TOKENS}" "0" "0.00" "0.00" "0" "crash" \
            "${COLD_CACHE}" "server failed to start" >> "${RESULTS_FILE}"
        continue
    fi

    for run in $(seq 1 "${RUNS_PER_K}"); do
        purge_cache

        echo -n "  run ${run}/${RUNS_PER_K}: "
        RESULT=$(run_single_bench "${MAX_TOKENS}" "${WARMUP_TOKENS}")
        echo "${RESULT}"

        # Parse result: tok_s \t ttft_s \t tokens \t quality
        TOK_S=$(echo "${RESULT}" | cut -f1)
        TTFT_S=$(echo "${RESULT}" | cut -f2)
        TOKENS_OUT=$(echo "${RESULT}" | cut -f3)
        QUALITY=$(echo "${RESULT}" | cut -f4)

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${TIMESTAMP}" "${TAG}" "${GIT_SHA}" "${CHIP}" "${GPU_CORES}" "${MEM_GB}" \
            "${OS_VER}" "${K}" "${MAX_TOKENS}" "${run}" "${TOK_S}" "${TTFT_S}" \
            "${TOKENS_OUT}" "${QUALITY}" "${COLD_CACHE}" "" >> "${RESULTS_FILE}"

        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    done

    cleanup
    sleep 1
done

echo ""
echo "============================================================"
echo " Done: ${TOTAL_RUNS} runs logged to results.tsv"
echo "============================================================"

# Print summary table
echo ""
echo "Summary (from this batch, tag=${TAG}):"
echo ""
printf "%-6s %-8s %-8s %-8s %-8s\n" "K" "tok/s" "ttft" "tokens" "quality"
printf "%-6s %-8s %-8s %-8s %-8s\n" "---" "------" "------" "------" "-------"

grep "	${TAG}	" "${RESULTS_FILE}" | grep "	${GIT_SHA}	" | while IFS=$'\t' read -r ts tag sha chip gpu mem os k tok run toks ttft tokout qual cold notes; do
    printf "%-6s %-8s %-8s %-8s %-8s\n" "${k}" "${toks}" "${ttft}" "${tokout}" "${qual}"
done
