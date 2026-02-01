#!/usr/bin/env python
"""Quick start script for the Meridian server."""

import subprocess
import sys
import time

def main():
    print("🚀 Starting Meridian ML Platform...")
    print("=" * 50)

    # Run server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "meridian.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]

    print(f"Command: {' '.join(cmd)}")
    print()
    print("📍 Server will be available at:")
    print("   - API:     http://127.0.0.1:8000")
    print("   - Swagger: http://127.0.0.1:8000/docs")
    print("   - ReDoc:   http://127.0.0.1:8000/redoc")
    print("   - Health:  http://127.0.0.1:8000/health")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 50)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()

