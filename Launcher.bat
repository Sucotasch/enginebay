@echo off
REM Запуск GUI лаунчера (fallback — используйте Launcher.vbs для запуска без консоли)
cd /d "%~dp0"
start /b pythonw launcher.py
