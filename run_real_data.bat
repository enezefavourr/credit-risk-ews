@echo off
REM One-click run on Windows: sets up a virtual environment, installs packages,
REM and runs the full pipeline on the Kaggle files in the PARENT folder of this repo.
REM Usage: double-click, or run  run_real_data.bat  from a terminal in this folder.

cd /d "%~dp0"
set "DATA_DIR=%~dp0.."
set "MPLBACKEND=Agg"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat

echo Installing packages (first run only takes a few minutes)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt || (echo Package install failed. & pause & exit /b 1)

echo.
echo Running pipeline on data in %DATA_DIR% ...
python run_all.py > run_real.log 2>&1
type run_real.log
python -m pytest -q tests >> run_real.log 2>&1

echo.
echo Done. Results are in the reports\ and powerbi\data\ folders; full log in run_real.log
pause
