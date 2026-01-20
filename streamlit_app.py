"""
Punto de entrada para Vercel - Streamlit App
"""
import subprocess
import sys

# Ejecutar el dashboard
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"],
    cwd="/home/sandro/Dev/Projects/portfolio-projects/market-price-ml"
)
