🤗 Hugging Face Streamlit App - Troubleshooting Guide
==================================================

✅ CURRENT STATUS: App is running on port 8501!
🌐 URL: http://localhost:8501

📋 MULTIPLE WAYS TO RUN THE APP:

1. 🎯 RECOMMENDED (Direct UV command):
   uv run streamlit run streamlit_app.py --server.port 8501

2. 🔧 Using the improved launcher:
   uv run python run_app.py

3. 💻 Using the simple launcher:
   uv run python simple_launcher.py

4. 📁 Using the batch file (Windows):
   Double-click run_app.bat

5. 🖱️ Manual Streamlit command:
   streamlit run streamlit_app.py

🔍 TROUBLESHOOTING STEPS:

❌ If "port already in use":
   • Check: netstat -an | findstr :8501
   • Kill existing: tasklist | findstr streamlit
   • Use different port: --server.port 8502

❌ If "command not found":
   • Install dependencies: uv sync
   • Check Streamlit: uv run streamlit --version

❌ If "module not found":
   • Reinstall: uv add streamlit plotly pandas numpy
   • Check environment: uv run python -c "import streamlit"

❌ If app won't load:
   • Try: http://127.0.0.1:8501
   • Try: http://localhost:8501
   • Check browser console for errors

🎛️ ALTERNATIVE PORTS:
   If 8501 is busy, try:
   • --server.port 8502
   • --server.port 8503
   • --server.port 3000

🌟 QUICK TEST:
   1. Open browser to: http://localhost:8501
   2. You should see the Hugging Face Showcase homepage
   3. Try the different pages in the sidebar

🆘 EMERGENCY COMMANDS:
   • Stop all Streamlit: taskkill /f /im streamlit.exe
   • Check what's running: netstat -an | findstr :85
   • Reset environment: uv sync --reinstall

💡 TIPS:
   • The app auto-reloads when you edit files
   • Use Ctrl+C to stop the server
   • Check the terminal for error messages
   • Models download on first use (requires internet)

🎉 SUCCESS INDICATORS:
   ✅ Terminal shows: "You can now view your Streamlit app in your browser"
   ✅ URL shows: "Local URL: http://localhost:8501"
   ✅ Browser loads the Hugging Face Showcase homepage
   ✅ Sidebar navigation works
   ✅ Models load without errors
