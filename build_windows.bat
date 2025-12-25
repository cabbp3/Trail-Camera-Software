@echo off
REM Trail Camera Organizer - Windows Build Script
REM Run this script on a Windows computer to build the executable
REM
REM REQUIREMENTS:
REM   - Python 3.9+ (with "Add to PATH" checked during install)
REM   - No C++ Build Tools needed (supabase replaced with REST API)
REM
REM OUTPUT:
REM   dist\TrailCamOrganizer\TrailCamOrganizer.exe

echo ========================================
echo Trail Camera Organizer - Windows Build
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

REM Convert icon to .ico if needed (skip if already exists)
if not exist "icon.ico" (
    echo Note: icon.ico not found. Building without custom icon.
    echo To add an icon, convert icon.png to icon.ico using an online converter.
)

REM Build the executable
echo.
echo Building executable...
pyinstaller TrailCamOrganizer_Windows.spec --clean

if errorlevel 1 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Executable location:
echo   dist\TrailCamOrganizer\TrailCamOrganizer.exe
echo.
echo To distribute:
echo   1. Copy the entire 'dist\TrailCamOrganizer' folder
echo   2. Users run TrailCamOrganizer.exe inside that folder
echo.
pause
