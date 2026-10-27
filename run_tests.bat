@echo off
echo ========================================================
echo   Running Pytest Suite (35 Unit Tests ^& Coverage Report)
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
pytest -v --cov=src --cov-report=term-missing
pause
