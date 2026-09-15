@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" python -m venv ".venv"
call ".venv\Scripts\activate.bat"
python -m pip install -r "requirements.txt"
if not exist "..\models" mkdir "..\models"
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ERROR: ffmpeg is required and was not found on PATH.
  echo Install it from https://ffmpeg.org/download.html and reopen this terminal.
  exit /b 1
)
uvicorn app.main:app --reload --port 8000
