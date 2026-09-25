#!/usr/bin/env bash
# Run the eval prompt set at several K values.
#
# One server per K: infer.m takes K only at launch, and its accept loop is
# serial, so each K gets its own process and they never overlap.
#
# Usage:
#   ./eval/sweep.sh              # K = 3 4 6 8
#   ./eval/sweep.sh 4 8          # only those
#   MODEL_DIR=... ./eval/sweep.sh
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_MODEL_DIR="$REPO/models/Qwen3.5-35B-A3B-4bit"
# Honour MODEL_DIR, but fall back to the in-repo model if it points nowhere --
# a stale MODEL_DIR exported for another checkout should not block the sweep.
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
if [ ! -d "$MODEL_DIR" ]; then
    if [ -d "$DEFAULT_MODEL_DIR" ]; then
        echo "WARNING: MODEL_DIR=$MODEL_DIR does not exist; using $DEFAULT_MODEL_DIR" >&2
        MODEL_DIR="$DEFAULT_MODEL_DIR"
    fi
fi
WEIGHTS="${WEIGHTS:-$REPO/metal_infer/out_35b/model_weights.bin}"
MANIFEST="${MANIFEST:-$REPO/metal_infer/out_35b/model_weights.json}"
VOCAB="${VOCAB:-$REPO/metal_infer/vocab.bin}"
INFER="$REPO/metal_infer/infer"
OUT="${OUT:-$REPO/eval/results}"
PORT="${PORT:-8300}"
# Thinking is off by default. Measured on this prompt set it changes no verdict
# and costs 78% more tokens, and leaving it on created a subtle failure: the
# --think-budget force-closes </think> mid-thought, the model keeps reasoning,
# and that continuation gets graded as the answer. Set THINKING=on to restore
# it (THINK_BUDGET then applies).
THINKING="${THINKING:-off}"
THINK_BUDGET="${THINK_BUDGET:-384}"
if [ "$THINKING" = "off" ]; then
    THINK_ARGS=(--no-think)
else
    THINK_ARGS=(-B "$THINK_BUDGET")
fi
KS=("$@"); [ ${#KS[@]} -eq 0 ] && KS=(3 4 6 8)

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

for f in "$INFER" "$WEIGHTS" "$MANIFEST" "$VOCAB"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
[ -d "$MODEL_DIR" ] || { echo "ERROR: missing model dir $MODEL_DIR" >&2; exit 1; }

mkdir -p "$OUT"

for K in "${KS[@]}"; do
    echo "=============== K=$K ==============="
    "$INFER" --model "$MODEL_DIR" --weights "$WEIGHTS" --manifest "$MANIFEST" \
             --vocab "$VOCAB" --k "$K" "${THINK_ARGS[@]}" --serve "$PORT" \
             > "$OUT/k$K.server.log" 2>&1 &
    SERVER_PID=$!

    ready=0
    for _ in $(seq 1 90); do
        if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "ERROR: server for K=$K never became ready (see $OUT/k$K.server.log)" >&2
        cleanup; SERVER_PID=""; continue
    fi

    python3 "$REPO/eval/run_eval.py" --port "$PORT" --k "$K" --out "$OUT"

    cleanup; SERVER_PID=""
    sleep 2   # let the port and page cache settle before the next K
done

echo
python3 "$REPO/eval/compare.py" --dir "$OUT"
