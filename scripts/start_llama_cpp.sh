#!/usr/bin/env bash
# ==========================================================================
#  llama-server launcher — Qwen3.6-27B IQ4_XS, OPTIMAL 96K config
#  Run: bash scripts/start_llama_cpp.sh
# ==========================================================================
set -euo pipefail

# --- Config (verified 2026-06-28) ---
LLAMA_SERVER="/c/Users/sucot/.cache/lm-studio/extensions/backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.20.1/llama-server.exe"
MODEL_GGUF="G:/Ai/Models/Qwen3.6-27B.i1-IQ4/Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"
HOST="127.0.0.1"
PORT="8080"
CTX_SIZE="98304"
GPU_LAYERS="99"
BATCH_SIZE="2048"
UBATCH_SIZE="512"
PARALLEL="1"
KV_UNIFIED="true"
CACHE_TYPE_K="q4_0"
CACHE_TYPE_V="q4_0"
THREADS="5"
FLASH_ATTN="on"
REASONING="off"
JINJA="true"
TEMPERATURE="1.0"
MIN_P="0.05"
TOP_P="0.95"
TOP_K="64"
MODEL_NAME="Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"

echo "============================================================"
echo " llama-server — $MODEL_NAME (OPTIMAL 96K config)"
echo " Model:  $MODEL_GGUF"
echo " GPU:    all layers (ngl=$GPU_LAYERS)"
echo " CTX:    $CTX_SIZE (96K)"
echo " KV:     unified, $CACHE_TYPE_K / $CACHE_TYPE_V"
echo " Batch:  $BATCH_SIZE / micro $UBATCH_SIZE"
echo " Listen: $HOST:$PORT"
echo " Speed:  ~34 tok/s decode, ~58 tok/s prefill"
echo "============================================================"
echo

# --- Find llama-server ---
if [[ ! -f "$LLAMA_SERVER" ]]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Install llama.cpp: https://github.com/ggml-org/llama.cpp/releases"
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
    --kv-unified \
    --cache-type-k "$CACHE_TYPE_K" \
    --cache-type-v "$CACHE_TYPE_V" \
    -t "$THREADS" \
    --flash-attn "$FLASH_ATTN" \
    --reasoning "$REASONING" \
    --temp "$TEMPERATURE" \
    --min-p "$MIN_P" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K"
