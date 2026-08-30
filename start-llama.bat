@echo off
REM ==========================================================================
REM  Start llama-server (ik_llama.cpp) — Qwen3.8-27B IQ4_KT/KS, 96K context
REM  Usage: start-llama.bat
REM
REM  Portability: paths come from env vars (with project-local defaults).
REM    LLAMA_SERVER  -> path to llama-server.exe (default: project ik_llama build)
REM    MODEL_GGUF    -> path to the GGUF model (set this to your model)
REM  You can also set them persistently:  setx LLAMA_SERVER "C:\...\llama-server.exe"
REM ==========================================================================
setlocal

set "SCRIPT_DIR=%~dp0"

REM --- Engine binary: prefer env, else the project ik_llama build ---
if defined LLAMA_SERVER (
    set "LLAMA_SERVER=%LLAMA_SERVER%"
) else (
    set "LLAMA_SERVER=%SCRIPT_DIR%ik_llama.cpp\versions\15dddc6\llama-server.exe"
)

REM --- Model: REQUIRED — point at your GGUF (no hardcoded default in repo) ---
if defined MODEL_GGUF (
    set "MODEL_GGUF=%MODEL_GGUF%"
) else (
    echo [ERROR] MODEL_GGUF is not set. Run:
    echo     set MODEL_GGUF=G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf
    echo     start-llama.bat
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
set "MODEL_NAME=%MODEL_GGUF:\=%"

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
    echo Build ik_llama.cpp via build-ikllama.bat, or use the launcher GUI
    echo (Engine -^> ik_llama.cpp) and set LLAMA_SERVER to your binary.
    pause
    exit /b 1
)

echo ============================================================
echo  Starting llama-server (ik_llama.cpp)...
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
    --cache-type-k %CACHE_TYPE_K% ^
    --cache-type-v %CACHE_TYPE_V% ^
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
