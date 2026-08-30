@echo off
REM ==========================================================================
REM  Start beellama.cpp llama-server — Qwen3.8-27B, 96K context, KVarN cache
REM  BeeLlama fork features: KVarN KV cache + precision tail (--kv-tail-tokens)
REM  Usage: start-beellama.bat
REM
REM  Portability: paths come from env vars (with project-local defaults).
REM    LLAMA_SERVER  -> path to a BeeLlama llama-server.exe (default: project beellama build)
REM    MODEL_GGUF    -> path to the GGUF model (set this to your model)
REM ==========================================================================
setlocal

set "SCRIPT_DIR=%~dp0"

REM --- BeeLlama binary: prefer env, else the project beellama build ---
if defined LLAMA_SERVER (
    set "LLAMA_SERVER=%LLAMA_SERVER%"
) else (
    set "LLAMA_SERVER=%SCRIPT_DIR%beellama.cpp\versions\v0.4.3-cuda-13.3\llama-server.exe"
)

REM --- Model: REQUIRED — point at your GGUF (no hardcoded default in repo) ---
if defined MODEL_GGUF (
    set "MODEL_GGUF=%MODEL_GGUF%"
) else (
    echo [ERROR] MODEL_GGUF is not set. Run:
    echo     set MODEL_GGUF=G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf
    echo     start-beellama.bat
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
set "CACHE_TYPE_K=kvarn5"
set "CACHE_TYPE_V=kvarn4"
set "KV_TAIL_TOKENS=1024"
set "THREADS=5"
set "MODEL_NAME=%MODEL_GGUF:\=%"

REM --- Check if already running ---
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] llama-server already running on %HOST%:%PORT%
    goto :done
)

REM --- Check binary ---
if not exist "%LLAMA_SERVER%" (
    echo [ERROR] BeeLlama llama-server not found at:
    echo   %LLAMA_SERVER%
    echo.
    echo Install a BeeLlama version via the launcher GUI:
    echo   Engine -^> BeeLlama.cpp -^> Check Updates -^> Download
    echo Repo: https://github.com/Anbeeld/beellama.cpp/releases
    pause
    exit /b 1
)

echo ============================================================
echo  Starting beellama.cpp llama-server...
echo  Model:  %MODEL_NAME%
echo  CTX:    %CTX_SIZE% (96K)
echo  KV:     %CACHE_TYPE_K% / %CACHE_TYPE_V%, tail %KV_TAIL_TOKENS%
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
    --kv-tail-tokens %KV_TAIL_TOKENS% ^
    -t %THREADS% ^
    --flash-attn on ^
    --reasoning auto ^
    --jinja ^
    --temp 1.0 ^
    --min-p 0.0 ^
    --top-p 0.95 ^
    --top-k 20

REM --- Wait for health ---
echo Waiting for server to load model...
set /a "attempt=0"
:wait_loop
set /a "attempt+=1"
if %attempt% gtr 120 (
    echo [ERROR] Server did not start in 240 seconds
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% neq 0 goto :wait_loop

echo [OK] llama-server ready on http://%HOST%:%PORT%
echo.

:done
