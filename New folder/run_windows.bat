@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=venv\.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Project virtual environment was not found.
    echo Expected either .venv\Scripts\python.exe or venv\.venv\Scripts\python.exe
    echo Create it with: py -m venv .venv
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m streamlit run app.py
pause
