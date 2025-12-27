@echo off
echo ============================================
echo Trail Camera Organizer - Windows Build
echo ============================================
echo.

cd /d "%~dp0"

echo Step 1: Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Please make sure Python 3.8+ is installed from python.org
        pause
        exit /b 1
    )
)

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 3: Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo Step 4: Building executable with PyInstaller...
echo This may take 5-10 minutes...
echo.
python -m PyInstaller TrailCamOrganizer_Windows.spec --clean

if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo BUILD COMPLETE!
echo ============================================
echo.
echo The app is located at:
echo   dist\TrailCamOrganizer\TrailCamOrganizer.exe
echo.

choice /C YN /M "Create a desktop shortcut now"
if errorlevel 2 goto :done
if errorlevel 1 goto :shortcut

:shortcut
echo Creating desktop shortcut...
cscript //nologo "Create Desktop Shortcut.vbs"
goto :done

:done
echo.
echo You can now run Trail Camera Organizer from the desktop!
echo.
pause
