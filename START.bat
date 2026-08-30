@echo off
REM ==========================================================================
REM  Quick launch: start server (ik_llama) + Hermes with local Qwen3.8-27B
REM  Double-click this file to start
REM
REM  Portability: MODEL_GGUF / LLAMA_SERVER come from env vars. Set them once:
REM    setx MODEL_GGUF "G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"
REM    setx LLAMA_SERVER "D:\path\to\llama-server.exe"
REM ==========================================================================
setlocal

set "HOST=127.0.0.1"
set "PORT=8080"

REM --- Model: REQUIRED ---
if not defined MODEL_GGUF (
    echo [ERROR] MODEL_GGUF is not set. Example:
    echo     setx MODEL_GGUF "G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf"
    pause
    exit /b 1
)

REM --- Binary: env or project ik_llama build ---
if not defined LLAMA_SERVER set "LLAMA_SERVER=%~dp0ik_llama.cpp\versions\15dddc6\llama-server.exe"
if not exist "%LLAMA_SERVER%" (
    echo [ERROR] llama-server not found: %LLAMA_SERVER%
    echo Set LLAMA_SERVER to your binary, or build via build-ikllama.bat
    pause
    exit /b 1
)

REM --- Check if server already running ---
curl -s http://%HOST%:%PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Server already running
    goto :launch
)

REM --- Start server ---
echo Starting llama-server...
start "" "%LLAMA_SERVER%" -m "%MODEL_GGUF%" -c 98304 -np 1 -ngl 99 -b 1024 -ub 256 --cache-type-k q4_0 --cache-type-v q4_0 -t 5 --flash-attn on --reasoning auto --jinja --temp 1.0 --min-p 0.0 --top-p 0.95 --top-k 20 --host %HOST% --port %PORT%

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
echo  Hermes + Qwen3.8-27B (96K context, ~35.9 tok/s)
echo  Server: http://%HOST%:%PORT%/v1
echo ============================================================
echo.
echo Use: hermes --provider local-llama --model <any-model-name>
echo.
pause
