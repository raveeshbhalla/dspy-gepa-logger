#!/usr/bin/env python3
"""
Check if the GEPA Observable dashboard is running on port 3000.

Usage:
    python check-dashboard.py [--start]

Options:
    --start    Attempt to start the dashboard if not running
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def check_port(host: str = "localhost", port: int = 3000, timeout: float = 2.0) -> bool:
    """Check if a server is responding on the given port."""
    url = f"http://{host}:{port}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return False


def find_web_directory() -> Path | None:
    """Find the web directory relative to current location."""
    # Try common locations
    candidates = [
        Path("web"),
        Path("../web"),
        Path("../../web"),
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "package.json").exists():
            return candidate.resolve()

    return None


def start_dashboard(web_dir: Path) -> bool:
    """Start the dashboard in the background."""
    print(f"Starting dashboard from {web_dir}...")

    try:
        # Start npm run dev in background
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=web_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait for it to start
        print("Waiting for dashboard to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_port():
                print(f"Dashboard started! (PID: {process.pid})")
                return True
            print(f"  Still waiting... ({i+1}s)")

        print("Dashboard did not start within 30 seconds")
        return False

    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js.")
        return False
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return False


def main():
    start_if_not_running = "--start" in sys.argv

    print("Checking if dashboard is running on http://localhost:3000...")

    if check_port():
        print()
        print("[OK] Dashboard is running!")
        print("     URL: http://localhost:3000")
        sys.exit(0)

    print()
    print("[NOT RUNNING] Dashboard is not responding on port 3000")

    if not start_if_not_running:
        print()
        print("To start the dashboard manually:")
        print("  cd web")
        print("  npm run dev")
        print()
        print("Or run this script with --start to auto-start:")
        print("  python check-dashboard.py --start")
        sys.exit(1)

    # Try to start it
    print()
    web_dir = find_web_directory()

    if not web_dir:
        print("Error: Could not find web directory")
        print("Make sure you're in the gepa-observable repository")
        sys.exit(1)

    if start_dashboard(web_dir):
        print()
        print("[OK] Dashboard is now running!")
        print("     URL: http://localhost:3000")
        sys.exit(0)
    else:
        print()
        print("[FAILED] Could not start dashboard")
        print("Please start it manually:")
        print(f"  cd {web_dir}")
        print("  npm run dev")
        sys.exit(1)


if __name__ == "__main__":
    main()
