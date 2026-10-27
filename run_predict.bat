@echo off
echo ========================================================
echo   Running Fraud Detection CLI Prediction Demo
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate

echo.
echo [1/2] Running Single Transaction Inference (Fraud Preset)...
python predict.py --demo-single fraud

echo.
echo [2/2] Running Batch CSV Screening (200 Transactions)...
python predict.py --batch data/batch_test_unlabelled.csv --output data/predictions_output.csv

echo.
echo ========================================================
echo   Prediction Demo Completed! Results saved to data/predictions_output.csv
echo ========================================================
pause
