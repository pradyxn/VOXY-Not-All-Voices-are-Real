@echo off
cd /d "%~dp0"
call npm install
npm run dev
