@echo off
setlocal

REM Define the vcpkg toolchain path
set VCPKG_TOOLCHAIN="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/vcpkg/scripts/buildsystems/vcpkg.cmake"

echo ==========================================
echo       NvidiaAIDenoiser Build Script
echo ==========================================

REM Create and enter build directory
if not exist build mkdir build
cd build

REM 1. Configure
echo.
echo [1/4] Configuring with CMake...
cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN%
if %errorlevel% neq 0 (
    echo Error: Configuration failed.
    pause
    exit /b %errorlevel%
)

REM 2. Build
echo.
echo [2/4] Building Release configuration...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Error: Build failed.
    pause
    exit /b %errorlevel%
)

REM 3. Install
echo.
echo [3/4] Installing to dist folder...
cmake --install . --config Release --prefix "../dist"
if %errorlevel% neq 0 (
    echo Error: Install failed.
    pause
    exit /b %errorlevel%
)

REM 4. Copy DLLs
echo.
echo [4/4] Copying runtime DLLs...
if exist "vcpkg_installed\x64-windows\bin" (
    xcopy "vcpkg_installed\x64-windows\bin\*.dll" "..\dist\bin" /Y /I
) else (
    echo Warning: vcpkg_installed bin directory not found. DLLs might be missing.
)

echo.
echo ==========================================
echo  SUCCESS: Distribution ready in dist\bin
echo ==========================================
pause
