#!/usr/bin/env python3
"""FastAPI server wrapping Project Adam's CognitiveAgent."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Project Adam API", version="1.0.0")

# lazy import to suppress startup noise
_agent = None
def get_agent():
    global _agent
    if _agent is None:
        import warnings; warnings.filterwarnings("ignore")
        from adam_chat import CognitiveAgent
        _agent = CognitiveAgent()
        print("[api] Agent ready")
    return _agent

class ChatRequest(BaseModel):
    message: str
    user_id: str = ""

class ChatResponse(BaseModel):
    reply: str
    user_id: str = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        agent = get_agent()
        reply = agent.chat(req.message)
        user = getattr(agent, "_current_user", None)
        return ChatResponse(reply=reply, user_id=user or req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
