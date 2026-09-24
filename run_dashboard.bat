@echo off
echo ============================================
echo  Credit Risk AI - Streamlit Dashboard
echo  http://localhost:8501
echo ============================================
cd /d "%~dp0"
python -m streamlit run frontend/app.py --server.port 8501 --server.address localhost
pause
