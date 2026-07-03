@echo off
REM ==========================================================================
REM  Stop llama-server
REM  Usage: stop-llama.bat
REM ==========================================================================
echo Stopping llama-server...
taskkill /F /IM llama-server.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] llama-server stopped
) else (
    echo [OK] llama-server was not running
)
