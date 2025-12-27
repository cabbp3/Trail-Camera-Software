@echo off
cd /d "%~dp0"

if exist "dist\TrailCamOrganizer\TrailCamOrganizer.exe" (
    start "" "dist\TrailCamOrganizer\TrailCamOrganizer.exe"
) else (
    echo Trail Camera Organizer has not been built yet.
    echo.
    echo Please run BUILD.bat first to create the executable.
    echo.
    pause
)
