#!/usr/bin/env bash
# ==========================================================================
#  llama-server launcher (ik_llama.cpp) — Qwen3.8-27B, OPTIMAL 96K config
#  Run: bash scripts/start_llama_cpp.sh
#
#  Portability: MODEL_GGUF / LLAMA_SERVER come from env vars.
#    export MODEL_GGUF="G:/Ai/Models/Qwen3.8-27B_qkv-IQ4_KS-MTP/Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"
#    export LLAMA_SERVER="D:/path/to/llama-server.exe"
# ==========================================================================
set -euo pipefail

# --- Config (ik_llama, Qwen3.8-27B, 96K pure, 35.9 t/s) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$SCRIPT_DIR/../ik_llama.cpp/versions/15dddc6/llama-server.exe}"
if [[ -z "${MODEL_GGUF:-}" ]]; then
    echo "ERROR: MODEL_GGUF is not set. Example:"
    echo '  export MODEL_GGUF="G:/Ai/Models/Qwen3.8-27B_qkv-IQ4_KS-MTP/Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"'
    exit 1
fi
HOST="127.0.0.1"
PORT="8080"
CTX_SIZE="98304"
GPU_LAYERS="99"
BATCH_SIZE="1024"
UBATCH_SIZE="256"
PARALLEL="1"
CACHE_TYPE_K="q4_0"
CACHE_TYPE_V="q4_0"
THREADS="5"
FLASH_ATTN="on"
REASONING="auto"
JINJA="true"
TEMPERATURE="1.0"
MIN_P="0.0"
TOP_P="0.95"
TOP_K="20"
MODEL_NAME="Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"

echo "============================================================"
echo " llama-server (ik_llama.cpp) — $MODEL_NAME (OPTIMAL 96K config)"
echo " Model:  $MODEL_GGUF"
echo " GPU:    all layers (ngl=$GPU_LAYERS)"
echo " CTX:    $CTX_SIZE (96K)"
echo " KV:     $CACHE_TYPE_K / $CACHE_TYPE_V"
echo " Batch:  $BATCH_SIZE / micro $UBATCH_SIZE"
echo " Listen: $HOST:$PORT"
echo " Speed:  ~35.9 tok/s decode, ~691 tok/s prefill (verified 2026-08)"
echo "============================================================"
echo

# --- Find llama-server ---
if [[ ! -f "$LLAMA_SERVER" ]]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build ik_llama.cpp via build-ikllama.bat, or set LLAMA_SERVER env var."
    exit 1
fi
echo "Using: $LLAMA_SERVER"
echo

# --- Launch ---
exec "$LLAMA_SERVER" \
    -m "$MODEL_GGUF" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX_SIZE" \
    -ngl "$GPU_LAYERS" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    --parallel "$PARALLEL" \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    -t "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --reasoning "$REASONING" \
    --jinja \
    --temp "$TEMPERATURE" \
    --min-p "$MIN_P" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K"
