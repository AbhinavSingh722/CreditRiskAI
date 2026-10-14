#!/usr/bin/env python3
"""
Credit Risk AI — One-click launcher
Starts FastAPI backend + opens Swagger docs in browser.
Usage: python run_api.py
"""
import os, sys, time, subprocess, webbrowser
from pathlib import Path

# ── Validate model bundle exists ──────────────────────────────
bundle = Path("models/pipeline_bundle.pkl")
if not bundle.exists():
    print("ERROR: models/pipeline_bundle.pkl not found.")
    print("Please run the notebook first: jupyter notebook AbhinavSingh_CreditRiskAI.ipynb")
    sys.exit(1)

os.environ["MODEL_PATH"] = str(bundle.resolve())

HOST = "127.0.0.1"
PORT = 8000

print("=" * 50)
print("  Credit Risk AI — FastAPI Backend")
print(f"  API:   http://{HOST}:{PORT}")
print(f"  Docs:  http://{HOST}:{PORT}/docs")
print("=" * 50)
print("Starting server... (Ctrl+C to stop)\n")

# Open docs in browser after a short delay
def open_browser():
    time.sleep(2)
    webbrowser.open(f"http://{HOST}:{PORT}/docs")

import threading
threading.Thread(target=open_browser, daemon=True).start()

# Start uvicorn
subprocess.run([
    sys.executable, "-m", "uvicorn", "backend.main:app",
    "--reload", "--host", HOST, "--port", str(PORT)
])
