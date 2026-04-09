@echo off
cd /d "%~dp0"
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Starting Motion Detection App...
python ultimate_motion_detector_v2.py
if errorlevel 1 (
    echo ERROR: Failed to start the application
    echo Make sure all dependencies are installed in the virtual environment
)
pause