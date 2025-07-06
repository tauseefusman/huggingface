#!/usr/bin/env python3
"""
Simple Streamlit Launcher
========================

Alternative launcher for the Hugging Face Streamlit app with better error handling.
"""

import os
import sys

def main():
    """Simple launcher that uses UV to run Streamlit"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        print("🚀 Starting Hugging Face Showcase App...")
        print("📱 App will open at: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Use UV to run streamlit
        os.system("uv run streamlit run streamlit_app.py --server.port 8501")
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure dependencies are installed: uv sync")
        print("2. Try running directly: uv run streamlit run streamlit_app.py")
        print("3. Check if port 8501 is already in use")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
