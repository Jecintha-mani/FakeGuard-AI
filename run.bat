@echo off
title FakeGuard AI - Setup & Run
color 0A

echo.
echo  =====================================================
echo    FakeGuard AI - Fake News Detector
echo  =====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not found in PATH!
    echo.
    echo  Please install Python from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found
python --version

echo.
echo  [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo  [OK] Dependencies installed

echo.
echo  [2/3] Training ML models (this may take 1-2 minutes)...
python train_model.py
if %errorlevel% neq 0 (
    echo  [ERROR] Model training failed!
    pause
    exit /b 1
)
echo  [OK] Models trained successfully

echo.
echo  [3/3] Starting web server...
echo.
echo  =====================================================
echo    Open your browser at: http://localhost:5000
echo    Press Ctrl+C to stop the server
echo  =====================================================
echo.
python app.py

pause
