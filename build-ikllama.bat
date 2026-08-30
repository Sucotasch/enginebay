@echo off
setlocal EnableExtensions
rem ============================================================================
rem  build-ikllama.bat — build ik_llama.cpp (ikawrakow fork) for Windows CUDA.
rem
rem  ik_llama.cpp has NO prebuilt binaries. This script builds llama-server.exe
rem  once from source, then drops it (plus the CUDA runtime DLLs) into
rem  ik_llama.cpp\versions\<commit>\, where the launcher GUI picks it up.
rem
rem  Requirements (already present on this machine):
rem    - Visual Studio 2022 Build Tools (cl.exe, vcvars64.bat)
rem    - CMake >= 3.24        - Git          - CUDA Toolkit (nvcc, cublas.lib)
rem
rem  Tune CUDA_ARCHITECTURES for your GPU: RTX 4070 Ti SUPER = "89" (Ada).
rem  NOTE: builds are slow on older CPUs (i7-5820K) — expect 30-90 min. Do not
rem  interrupt; a partial build forces a full restart from scratch.
rem ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SRC_DIR=%SCRIPT_DIR%ik_llama.cpp"
set "BUILD_DIR=%SRC_DIR%\build"

set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDA_PATH=%CUDA_DIR%"
set "CUDA_PATH_V13_1=%CUDA_DIR%"
set "CUDA_ARCH=89"

rem MSVC environment (cl.exe, link.exe) — required for CUDA + C++ builds
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

if not exist "%SRC_DIR%\.git" (
    echo [1/4] Cloning ik_llama.cpp...
    git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git "%SRC_DIR%"
    if errorlevel 1 ( echo FAILED: git clone & pause & exit /b 1 )
) else (
    echo [1/4] Repo already present, updating...
    git -C "%SRC_DIR%" pull --ff-only
)

echo [2/4] Configuring CMake (CUDA arch=%CUDA_ARCH%)...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES="%CUDA_ARCH%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_LLAMAFILE=OFF
if errorlevel 1 ( echo FAILED: cmake configure & pause & exit /b 1 )

echo [3/4] Building (this is the slow step, be patient)...
rem Parallel build: pass the job count as the first argument, default 6.
rem   build-ikllama.bat 8   -> 8 parallel cl.exe processes
rem IMPORTANT: run this .bat by hand (double-click / your own terminal).
rem If an agent launches it through a file sandbox, the sandbox blocks the
rem inter-process pipes MSBuild needs for parallel worker nodes, so the
rem build silently degrades to ONE process (11-13% CPU, several hours).
rem Outside the sandbox /m:N parallelizes normally (4-6 processes).
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=6"
cmake --build "%BUILD_DIR%" --config Release --parallel %JOBS%
if errorlevel 1 ( echo FAILED: cmake build & pause & exit /b 1 )

echo [4/4] Installing into versions folder...
for /f "delims=" %%i in ('git -C "%SRC_DIR%" rev-parse --short HEAD') do set "COMMIT=%%i"
set "VERSION_DIR=%SCRIPT_DIR%ik_llama.cpp\versions\%COMMIT%"
mkdir "%VERSION_DIR%" 2>nul

if exist "%BUILD_DIR%\bin\Release\llama-server.exe" (
    set "BIN_DIR=%BUILD_DIR%\bin\Release"
) else (
    set "BIN_DIR=%BUILD_DIR%\bin"
)
copy /y "%BIN_DIR%\llama-server.exe" "%VERSION_DIR%\" >nul
copy /y "%BIN_DIR%\*.dll" "%VERSION_DIR%\" >nul 2>nul

rem CUDA runtime DLLs are NOT in the toolkit bin folder on this machine;
rem copy them from the beellama install (same CUDA 13.1 runtime).
set "BEELAMA_DLL=%SCRIPT_DIR%beellama.cpp\versions\preview-v0.4.4-cuda-13.1"
if exist "%BEELAMA_DLL%\cublas64_13.dll" (
    copy /y "%BEELAMA_DLL%\cublas64_13.dll"   "%VERSION_DIR%\" >nul
    copy /y "%BEELAMA_DLL%\cublasLt64_13.dll" "%VERSION_DIR%\" >nul
    copy /y "%BEELAMA_DLL%\cudart64_13.dll"   "%VERSION_DIR%\" >nul
    echo     CUDA runtime DLLs copied from beellama folder
)

echo.
echo DONE. Installed into:
echo     %VERSION_DIR%
echo.
echo Now in the launcher: Engine = ik_llama.cpp, click "Use" on this version,
echo then pick the Qwen3.8-27B IQ4_KT/KS MTP model (it needs this engine).
pause
