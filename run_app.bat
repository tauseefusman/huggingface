@echo off
echo 🚀 Launching Hugging Face Streamlit App...
echo 📱 App will be available at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the server
echo ================================================

cd /d "%~dp0"

REM Try to run with UV first
uv run streamlit run streamlit_app.py --server.port 8501 --server.address localhost

REM If UV fails, try with regular Python
if errorlevel 1 (
    echo.
    echo ⚠️ UV failed, trying with regular Python...
    python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
)

pause
