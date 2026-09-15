@echo off
cd /d "%~dp0"
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ERROR: ffmpeg is required and was not found on PATH.
  echo Install it from https://ffmpeg.org/download.html and reopen this terminal.
  exit /b 1
)
if not exist "backend\.venv\Scripts\python.exe" call "backend\setup.bat"
start "VOXY Backend" cmd /k "cd /d \"%~dp0backend\" && call \".venv\Scripts\activate.bat\" && uvicorn app.main:app --reload --port 8000"
cd /d "%~dp0frontend"
if not exist "node_modules" call npm install
start "VOXY Frontend" cmd /k "cd /d \"%~dp0frontend\" && npm run dev"
timeout /t 3 /nobreak >nul
start http://localhost:3000
