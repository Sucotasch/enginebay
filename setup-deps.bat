@echo off
REM ==========================================================================
REM  EngineBay — one-time dependency setup for a fresh machine.
REM  Installs the Python packages needed by launcher.py and scripts/smoke_test.py.
REM  Usage: double-click setup-deps.bat (or run it from a terminal)
REM ==========================================================================
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://www.python.org/
    echo         and make sure "Add python.exe to PATH" is checked.
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check your network / Python version.
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed.
echo Now launch the GUI:  Launcher.vbs  (or  python launcher.py)
echo.
pause
