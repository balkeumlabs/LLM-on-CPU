@echo off
REM Convenience wrapper (Windows) for the colleague's Intel laptop.
REM Usage: run.bat [--report-only] [--model NAME] [--cpu-only]
cd /d "%~dp0"
REM Prefer the project venv if it exists, else system python.
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run_experiment.py %*
) else (
    where python >nul 2>nul && (python run_experiment.py %*) || (py run_experiment.py %*)
)
