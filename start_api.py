"""
start_api.py — launches the AudiobookMaker FastAPI backend with uvicorn.
"""
import os
import sys

import uvicorn

_HOST: str = "127.0.0.1"
_PORT: int = 8000
# Seconds uvicorn waits for open connections (WebSocket event streams, a
# running download) to finish after Ctrl+C / SIGTERM before it closes them.
# Without a limit one lingering client keeps the process, and the model it
# holds in VRAM, alive indefinitely.
_GRACEFUL_SHUTDOWN_SEC: int = 10

if __name__ == "__main__":
    # Ensure project root is in python path
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    print("=" * 60)
    print("           Launching AudiobookMaker FastAPI Server            ")
    print(f"                Host: {_HOST} | Port: {_PORT}                 ")
    print("=" * 60)

    uvicorn.run(
        "api.server:app",
        host=_HOST,
        port=_PORT,
        reload=False,
        timeout_graceful_shutdown=_GRACEFUL_SHUTDOWN_SEC,
    )
