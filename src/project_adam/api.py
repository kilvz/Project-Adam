import json
import logging
import asyncio
import queue
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from .agent import CognitiveAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="Project Adam API", version="1.0.0")

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
            _agent = CognitiveAgent()
        logger.info("Agent ready")
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
        user = agent._current_user if hasattr(agent, "_current_user") else None
        return ChatResponse(reply=reply, user_id=user or req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    agent = get_agent()
    q = queue.Queue()

    def token_callback(tok):
        q.put(tok)

    def generate():
        try:
            agent.chat(req.message, token_callback=token_callback)
            q.put(None)
        except Exception as e:
            q.put(e)

    import threading
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            try:
                tok = await asyncio.get_event_loop().run_in_executor(None, q.get)
            except Exception:
                break
            if tok is None:
                break
            if isinstance(tok, Exception):
                yield f"data: {json.dumps({'error': str(tok)})}\n\n"
                break
            yield f"data: {json.dumps({'token': tok})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/users")
def list_users():
    agent = get_agent()
    profiles = agent.user_profiles.profiles
    return [
        {
            "name": p["name"],
            "interaction_count": p.get("interaction_count", 0),
            "avg_sentiment": p.get("avg_sentiment", 0.0),
            "topics": list(p.get("topics", {}).keys()),
        }
        for p in profiles.values()
    ]


@app.get("/users/{name}")
def get_user(name: str):
    agent = get_agent()
    profile = agent.user_profiles.profiles.get(name.strip().capitalize())
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile


@app.get("/memory/episodic")
def search_episodic(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20),
):
    agent = get_agent()
    results = agent.episodic_memory.search(query, k=k)
    return [
        {"text": text, "similarity": round(sim, 4), "reward": reward}
        for text, sim, reward in results
    ]


@app.get("/memory/semantic")
def search_semantic(
    query: str = Query(..., description="Search query"),
    k: int = Query(3, ge=1, le=20),
):
    agent = get_agent()
    results = agent.semantic_memory.retrieve(query, k=k)
    return [
        {"category": cat, "facts": facts, "similarity": round(sim, 4)}
        for cat, facts, sim in results
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
