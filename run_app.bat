@echo off
echo ========================================================
echo   Launching Credit Card Fraud Detection Streamlit App
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py
pause
