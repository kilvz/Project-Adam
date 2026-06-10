#!/usr/bin/env python3
"""FastAPI server wrapping Project Adam's CognitiveAgent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from project_adam.api import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
