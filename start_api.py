import sys
import uvicorn
import os

if __name__ == "__main__":
    # Ensure project root is in python path
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
        
    print("=" * 60)
    print("           Launching AudiobookMaker FastAPI Server            ")
    print("                Host: 127.0.0.1 | Port: 8000                 ")
    print("=" * 60)
    
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
