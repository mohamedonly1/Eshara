@echo off
[ignoring loop detection]
chcp 65001 > nul
title Ishara Project - Auto Run

echo ==================================================
echo   Ishara Sign Language Recognition System Setup
echo ==================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10 from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Check if virtual environment (venv) folder exists
if not exist "venv\" (
    echo Creating virtual environment (venv)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

:: 3. Activate venv and install/verify dependencies
echo Verifying and installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies verified successfully.
echo.

:: 4. Start the server and open the browser
echo Starting Ishara server...
start http://localhost:5000/
venv\Scripts\python.exe server.py

if errorlevel 1 (
    echo [ERROR] Server terminated with an error.
    pause
)
