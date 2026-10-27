@echo off
echo ========================================================
echo   Training Anomaly Detection Models ^& Generating Artifacts
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
python train.py
pause
