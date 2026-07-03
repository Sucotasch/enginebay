@echo off
REM ==========================================================================
REM  Quick launch: start server + Hermes with local Qwen3.6-27B
REM  Double-click this file to start
REM ==========================================================================
setlocal

set "HOST=127.0.0.1"
set "PORT=8080"
set "LLAMA_SERVER=C:\Users\sucot\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.20.1\llama-server.exe"
set "MODEL=G:\Ai\Models\Qwen3.6-27B.i1-IQ4\Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf"

REM --- Check if server already running ---
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Server already running
    goto :launch
)

REM --- Start server ---
echo Starting llama-server...
start "" "%LLAMA_SERVER%" -m "%MODEL%" -c 98304 -ngl 99 -b 2048 -ub 512 --kv-unified --cache-type-k q4_0 --cache-type-v q4_0 -t 5 --flash-attn on --reasoning off --temp 1.0 --min-p 0.05 --top-p 0.95 --top-k 64 --host %HOST% --port %PORT%

REM --- Wait for server ---
echo Waiting for model to load...
:wait
timeout /t 2 /nobreak >nul
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% neq 0 goto :wait
echo [OK] Server ready!

:launch
REM --- Launch Hermes ---
echo.
echo ============================================================
echo  Hermes + Qwen3.6-27B (96K context, ~34 tok/s)
echo  Server: http://%HOST%:%PORT%/v1
echo ============================================================
echo.
echo Use: hermes --provider local-llama --model <any-model-name>
echo.
pause
