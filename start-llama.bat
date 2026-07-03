@echo off
REM ==========================================================================
REM  Start llama-server — Qwen3.6-27B IQ4_XS, 96K context
REM  Usage: start-llama.bat
REM ==========================================================================
setlocal

set "SCRIPT_DIR=%~dp0"
set "LLAMA_SERVER=C:\Users\sucot\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.20.1\llama-server.exe"
set "MODEL_GGUF=G:\Ai\Models\Qwen3.6-27B.i1-IQ4\Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"
set "HOST=127.0.0.1"
set "PORT=8080"
set "CTX_SIZE=98304"
set "GPU_LAYERS=99"
set "BATCH_SIZE=2048"
set "UBATCH_SIZE=512"
set "PARALLEL=1"
set "CACHE_TYPE_K=q4_0"
set "CACHE_TYPE_V=q4_0"
set "THREADS=5"
set "MODEL_NAME=Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"

REM --- Check if already running ---
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] llama-server already running on %HOST%:%PORT%
    goto :done
)

REM --- Check binary ---
if not exist "%LLAMA_SERVER%" (
    echo [ERROR] llama-server not found at:
    echo   %LLAMA_SERVER%
    echo.
    echo Install: https://github.com/ggml-org/llama.cpp/releases
    pause
    exit /b 1
)

echo ============================================================
echo  Starting llama-server...
echo  Model:  %MODEL_NAME%
echo  CTX:    %CTX_SIZE% (96K)
echo  Port:   %HOST%:%PORT%
echo ============================================================
echo.

start "" "%LLAMA_SERVER%" ^
    -m "%MODEL_GGUF%" ^
    --host %HOST% ^
    --port %PORT% ^
    -c %CTX_SIZE% ^
    -ngl %GPU_LAYERS% ^
    -b %BATCH_SIZE% ^
    -ub %UBATCH_SIZE% ^
    --parallel %PARALLEL% ^
    --kv-unified ^
    --cache-type-k %CACHE_TYPE_K% ^
    --cache-type-v %CACHE_TYPE_V% ^
    -t %THREADS% ^
    --flash-attn on ^
    --reasoning off ^
    --jinja ^
    --temp 1.0 ^
    --min-p 0.05 ^
    --top-p 0.95 ^
    --top-k 64

REM --- Wait for health ---
echo Waiting for server to load model...
set /a "attempt=0"
:wait_loop
set /a "attempt+=1"
if %attempt% gtr 60 (
    echo [ERROR] Server did not start in 60 seconds
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% neq 0 goto :wait_loop

echo [OK] llama-server ready on http://%HOST%:%PORT%
echo.

:done
