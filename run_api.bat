@echo off
echo Starting Patient Deterioration Prediction API...
echo API will be available at: http://localhost:8000
echo API Documentation at: http://localhost:8000/docs
echo.
python -m uvicorn inference.main:app --host 0.0.0.0 --port 8000 --reload
pause
