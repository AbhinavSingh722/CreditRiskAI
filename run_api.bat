@echo off
echo ============================================
echo  Credit Risk AI - FastAPI Backend
echo  http://localhost:8000/docs
echo ============================================
cd /d "%~dp0"
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
pause
