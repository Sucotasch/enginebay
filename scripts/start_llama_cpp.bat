@echo off
REM ==========================================================================
REM  llama-server launcher (ik_llama.cpp) — Qwen3.8-27B, OPTIMAL 96K config
REM  Run: scripts\start_llama_cpp.bat
REM
REM  Portability: MODEL_GGUF / LLAMA_SERVER come from env vars.
REM    set MODEL_GGUF=G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf
REM    set LLAMA_SERVER=D:\path\to\llama-server.exe
REM ==========================================================================
setlocal

REM --- Config (ik_llama, Qwen3.8-27B, 96K pure, 35.9 t/s) ---
if not defined LLAMA_SERVER set "LLAMA_SERVER=%~dp0..\ik_llama.cpp\versions\15dddc6\llama-server.exe"
if not defined MODEL_GGUF (
    echo [ERROR] MODEL_GGUF is not set. Example:
    echo     set MODEL_GGUF=G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf
    pause
    exit /b 1
)
set "HOST=127.0.0.1"
set "PORT=8080"
set "CTX_SIZE=98304"
set "GPU_LAYERS=99"
set "BATCH_SIZE=1024"
set "UBATCH_SIZE=256"
set "PARALLEL=1"
set "CACHE_TYPE_K=q4_0"
set "CACHE_TYPE_V=q4_0"
set "THREADS=5"
set "FLASH_ATTN=on"
set "REASONING=auto"
set "JINJA=true"
set "TEMPERATURE=1.0"
set "MIN_P=0.0"
set "TOP_P=0.95"
set "TOP_K=20"
set "MODEL_NAME=Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"

echo ============================================================
echo  llama-server (ik_llama.cpp) — %MODEL_NAME% (OPTIMAL 96K config)
echo  Model:  %MODEL_GGUF%
echo  GPU:    all layers (ngl=%GPU_LAYERS%)
echo  CTX:    %CTX_SIZE% (96K)
echo  KV:     %CACHE_TYPE_K% / %CACHE_TYPE_V%
echo  Batch:  %BATCH_SIZE% / micro %UBATCH_SIZE%
echo  Listen: %HOST%:%PORT%
echo  Speed:  ~35.9 tok/s decode, ~691 tok/s prefill (verified 2026-08)
echo ============================================================
echo.

if not exist "%LLAMA_SERVER%" (
    echo ERROR: llama-server not found at %LLAMA_SERVER%
    echo Build ik_llama.cpp via build-ikllama.bat, or set LLAMA_SERVER env var.
    pause
    exit /b 1
)
echo Using: %LLAMA_SERVER%
echo.

"%LLAMA_SERVER%" ^
    -m "%MODEL_GGUF%" ^
    --host %HOST% ^
    --port %PORT% ^
    -c %CTX_SIZE% ^
    -ngl %GPU_LAYERS% ^
    -b %BATCH_SIZE% ^
    -ub %UBATCH_SIZE% ^
    --parallel %PARALLEL% ^
    --cache-type-k %CACHE_TYPE_K% ^
    --cache-type-v %CACHE_TYPE_V% ^
    -t %THREADS% ^
    --flash-attn %FLASH_ATTN% ^
    --reasoning %REASONING% ^
    --jinja ^
    --temp %TEMPERATURE% ^
    --min-p %MIN_P% ^
    --top-p %TOP_P% ^
    --top-k %TOP_K%

pause