#!/usr/bin/env python3
"""
Streamlit App Launcher
=====================

Simple script to launch the Hugging Face Streamlit showcase app.
"""

import subprocess
import sys
import os
import socket
import webbrowser
import time

def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8501):
    """Find an available port starting from start_port"""
    port = start_port
    while port < start_port + 10:  # Try 10 ports
        if check_port_available(port):
            return port
        port += 1
    return None

def main():
    """Launch the Streamlit app"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Check for available port
        port = find_available_port(8501)
        if port is None:
            print("❌ No available ports found between 8501-8510")
            print("   Please check for running Streamlit instances")
            return 1
        
        # Launch streamlit
        print("🚀 Launching Hugging Face Showcase App...")
        print(f"📱 App will be available at: http://localhost:{port}")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Prepare the command
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "true"
        ]
        
        # Start the process
        process = subprocess.Popen(cmd)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"🌐 Opening browser at {url}")
        webbrowser.open(url)
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
