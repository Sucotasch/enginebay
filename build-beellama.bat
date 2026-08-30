@echo off
setlocal EnableExtensions

rem ============================================================================
rem  AUTO-LOG WRAPPER — re-runs this script once, teeing all output to a log
rem  file (so the console stays live AND a log is saved for debugging). The
rem  env-var guard prevents infinite recursion. Build failures propagate the
rem  exit code back to the caller.
rem ============================================================================
if defined BEELAMA_LOG_ACTIVE goto :real
set "BEELAMA_LOG_ACTIVE=1"
set "BEELAMA_LOG_FILE=%~dp0beellama-build.log"
echo [auto-log] Saving a copy of this run's output to %BEELAMA_LOG_FILE%
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '%~f0' %* 2>&1 | Tee-Object -FilePath '%BEELAMA_LOG_FILE%'; exit $LASTEXITCODE }"
exit /b %ERRORLEVEL%

:real
rem ============================================================================
rem  build-beellama.bat — build beellama.cpp (Anbeeld fork) for Windows CUDA.
rem
rem  Builds llama-server.exe from the preview-v0.4.4 tag (same code as the fast
rem  "preview-v0.4.4-cuda-13.1" binary that was deleted by accident, but built
rem  locally with MSVC + CUDA 13.1 instead of the LLVM + CUDA 13.3 toolchain
rem  that gave the ~19 t/s regression on v0.4.4-cuda-13.3).
rem
rem  Requirements (already present on this machine):
rem    - Visual Studio 2022 Build Tools (cl.exe, vcvars64.bat)
rem    - CMake >= 3.24        - Git          - CUDA Toolkit 13.1
rem
rem  Tune CUDA_ARCHITECTURES for your GPU: RTX 4070 Ti SUPER = "89" (Ada).
rem  NOTE: builds are slow on older CPUs (i7-5820K) — expect 30-90 min. Do not
rem  interrupt; a partial build forces a full restart from scratch.
rem ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SRC_DIR=%SCRIPT_DIR%beellama.cpp"
set "BUILD_DIR=%SRC_DIR%\build"

set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDA_PATH=%CUDA_DIR%"
set "CUDA_PATH_V13_1=%CUDA_DIR%"
set "CUDA_ARCH=89"

rem MSVC environment (cl.exe, link.exe) — required for CUDA + C++ builds
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

echo [1/4] Ensuring source is at preview-v0.4.4 tag...
git -C "%SRC_DIR%" checkout preview-v0.4.4 2>nul
if errorlevel 1 (
    echo FAILED: git checkout preview-v0.4.4 — is the tag missing?
    pause & exit /b 1
)

echo [2/4] Configuring CMake (CUDA 13.1, arch=%CUDA_ARCH%)...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCH%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_LLAMAFILE=OFF
if errorlevel 1 ( echo FAILED: cmake configure & pause & exit /b 1 )

echo [3/4] Building (this is the slow step, be patient)...
rem Parallel build: pass the job count as the first argument, default 6.
rem   build-beellama.bat 8   -> 8 parallel cl.exe processes
rem IMPORTANT: run this .bat by hand (double-click / your own terminal).
rem If an agent launches it through a file sandbox, the sandbox blocks the
rem inter-process pipes MSBuild needs for parallel worker nodes, so the
rem build silently degrades to ONE process (11-13% CPU, several hours).
rem Outside the sandbox /m:N parallelizes normally (4-6 processes).
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=6"
cmake --build "%BUILD_DIR%" --config Release --target llama-server --parallel %JOBS%
if errorlevel 1 ( echo FAILED: cmake build & pause & exit /b 1 )

echo [4/4] Installing into versions folder...
set "VERSION_DIR=%SCRIPT_DIR%beellama.cpp\versions\preview-v0.4.4-msvc"
mkdir "%VERSION_DIR%" 2>nul

if exist "%BUILD_DIR%\bin\Release\llama-server.exe" (
    set "BIN_DIR=%BUILD_DIR%\bin\Release"
) else (
    set "BIN_DIR=%BUILD_DIR%\bin"
)
copy /y "%BIN_DIR%\llama-server.exe" "%VERSION_DIR%\" >nul
copy /y "%BIN_DIR%\*.dll" "%VERSION_DIR%\" >nul 2>nul

rem CUDA runtime DLLs — copy from the existing v0.4.4-cuda-13.3 install
rem (same CUDA 13 major version, backward compatible with 13.1 build)
set "CUDA_DLL_DIR=%SCRIPT_DIR%beellama.cpp\versions\v0.4.4-cuda-13.3"
if exist "%CUDA_DLL_DIR%\cublas64_13.dll" (
    copy /y "%CUDA_DLL_DIR%\cublas64_13.dll"   "%VERSION_DIR%\" >nul
    copy /y "%CUDA_DLL_DIR%\cublasLt64_13.dll" "%VERSION_DIR%\" >nul
    copy /y "%CUDA_DLL_DIR%\cudart64_13.dll"   "%VERSION_DIR%\" >nul
    echo     CUDA runtime DLLs copied from v0.4.4-cuda-13.3
) else (
    echo     WARNING: CUDA runtime DLLs not found — copy them manually later
)

echo.
echo DONE. Installed into:
echo     %VERSION_DIR%
echo.
echo Now in the launcher: Engine = beellama.cpp, click "Use" on
echo "preview-v0.4.4-msvc", then test the pure preset.
echo.
echo Verification: this MSVC + CUDA 13.1 build should restore ~30 t/s
echo from the deleted preview-v0.4.4-cuda-13.1 binary.
pause