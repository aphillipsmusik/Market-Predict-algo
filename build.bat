@echo off
REM ============================================================================
REM  Build the SPY Predictor Windows desktop app.
REM
REM  Prerequisites:
REM    * Python 3.10+ on PATH (python --version should work)
REM    * Internet access (for pip + yfinance initial data pull)
REM
REM  Output: dist\SPYPredictor\SPYPredictor.exe
REM ============================================================================
setlocal enableextensions enabledelayedexpansion

echo.
echo === SPY Predictor — Windows build ===
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install from https://www.python.org/ first.
    exit /b 1
)

if not exist ".venv\" (
    echo Creating virtual environment .venv\
    python -m venv .venv || goto :error
)

call .venv\Scripts\activate.bat || goto :error

echo Installing / updating dependencies...
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error
python -m pip install pyinstaller || goto :error

echo.
echo Pre-training models so the app ships ready-to-use...
python -m scripts.train --no-lstm --no-backtest
if errorlevel 1 (
    echo WARNING: model pre-training failed — build will continue, but the app
    echo will train on first launch, which needs internet access.
)

echo.
echo Running PyInstaller...
python launcher\build.py || goto :error

echo.
echo === BUILD COMPLETE ===
echo Output: dist\SPYPredictor\SPYPredictor.exe
echo Zip or Inno-Setup the dist\SPYPredictor folder to distribute.
exit /b 0

:error
echo.
echo BUILD FAILED. See errors above.
exit /b 1
