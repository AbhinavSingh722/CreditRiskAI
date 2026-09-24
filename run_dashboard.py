#!/usr/bin/env python3
"""
Credit Risk AI — Streamlit Dashboard Launcher
Usage: python run_dashboard.py
"""
import os, sys, subprocess
from pathlib import Path

if not Path("models/pipeline_bundle.pkl").exists():
    print("ERROR: models/pipeline_bundle.pkl not found.")
    print("Run the notebook first, then start the API (run_api.py), then this.")
    sys.exit(1)

print("=" * 50)
print("  Credit Risk AI — Streamlit Dashboard")
print("  URL: http://localhost:8501")
print("=" * 50)
print("Make sure the API is running (python run_api.py)")
print("Starting dashboard... (Ctrl+C to stop)\n")

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "frontend/app.py",
    "--server.port", "8501",
    "--server.address", "localhost",
    "--server.headless", "false",
])
