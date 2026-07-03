@echo off
REM ==========================================================================
REM  Launch Hermes with local Qwen3.6-27B provider
REM  Usage: launch-hermes-llama.bat
REM  1. Starts llama-server (if not running)
REM  2. Waits for model to load
REM  3. Launches Hermes with local provider
REM ==========================================================================
setlocal

set "SCRIPT_DIR=%~dp0"
set "HOST=127.0.0.1"
set "PORT=8080"

REM --- Start server if not running ---
call "%SCRIPT_DIR%start-llama.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start llama-server
    pause
    exit /b 1
)

echo ============================================================
echo  Launching Hermes with local Qwen3.6-27B provider
echo  Server: http://%HOST%:%PORT%/v1
echo  Model:  Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf
echo ============================================================
echo.

REM --- Launch Hermes with local provider ---
set OPENAI_BASE_URL=http://%HOST%:%PORT%/v1
set OPENAI_API_KEY=not-needed

hermes --provider local-llama --model Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf

pause
