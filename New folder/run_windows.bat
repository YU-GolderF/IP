@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment was not found.
    echo Expected: .venv\Scripts\python.exe
    echo Create it with: py -m venv .venv
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py
pause
