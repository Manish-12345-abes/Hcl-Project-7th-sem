@echo off
echo ========================================================
echo   Running HCL Project Verification ^& Completeness Audit
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
python verify_project.py
pause
