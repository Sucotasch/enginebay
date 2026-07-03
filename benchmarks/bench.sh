#!/usr/bin/env bash
# ==========================================================================
#  Minimal benchmark — measure TTFT, TPOT, throughput for llama-server
#  Run: bash benchmarks/bench.sh [prompt_tokens] [output_tokens] [concurrency]
# ==========================================================================
set -euo pipefail

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL_NAME:-Qwen3.6-27B-IQ4_XS}"
PROMPT_TOKENS="${1:-512}"
OUTPUT_TOKENS="${2:-256}"
CONCURRENCY="${3:-1}"

echo "Benchmark: $MODEL"
echo "  Prompt: ~$PROMPT_TOKENS tokens, Output: $OUTPUT_TOKENS tokens, Concurrency: $CONCURRENCY"
echo "  Endpoint: $BASE_URL"
echo

# Generate a prompt of approximate length
PROMPT=$(python -c "print('hello ' * $PROMPT_TOKENS)")

echo "--- Single Request ---"
START=$(python -c "import time; print(time.time())")
RESPONSE=$(curl -s "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": $(python -c "import json; print(json.dumps('$PROMPT'[:2000] + ' Reply in exactly 100 words.'))")}],
        \"max_tokens\": $OUTPUT_TOKENS,
        \"temperature\": 0
    }")
END=$(python -c "import time; print(time.time())")

ELAPSED=$(python -c "print(f'{$END - $START:.2f}')")
echo "  Time: ${ELAPSED}s"
echo "  Response (first 200 chars):"
echo "$RESPONSE" | python -c "import sys,json; r=json.load(sys.stdin); print('  ', r['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "  (parse error)"

echo
echo "--- Streaming TTFT Test ---"
START=$(python -c "import time; print(time.time())")
FIRST_TOKEN=$(curl -sN "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
        \"max_tokens\": 64,
        \"temperature\": 0,
        \"stream\": true
    }" | head -c 500)
TTFT=$(python -c "import time; print(f'{time.time() - $START:.2f}')")
echo "  TTFT: ${TTFT}s (approximate, first 500 bytes of stream)"

echo
echo "Done."
