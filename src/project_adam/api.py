import json
import time
import logging
import asyncio
import queue
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from .agent import CognitiveAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="Project Adam API", version="1.0.0")

_agent = None

MODEL_ID = "adam-cognet"


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


# ── OpenAI-compatible endpoints ─────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 128

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]


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


# ── OpenAI-compatible endpoints ─────────────────────────────────────

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "project_adam",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    agent = get_agent()
    user_text = req.messages[-1].content if req.messages else ""

    if req.stream:
        return _chat_completions_stream(agent, user_text, req)

    reply = agent.chat(user_text)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
    )


def _chat_completions_stream(agent, user_text, req):
    q = queue.Queue()

    def token_callback(tok):
        q.put(tok)

    def generate():
        try:
            agent.chat(user_text, token_callback=token_callback)
            q.put(None)
        except Exception as e:
            q.put(e)

    import threading
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_stream():
        yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        while True:
            try:
                tok = await asyncio.get_event_loop().run_in_executor(None, q.get)
            except Exception:
                break
            if tok is None:
                yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                break
            if isinstance(tok, Exception):
                yield f"data: {json.dumps({'error': {'message': str(tok)}})}\n\n"
                break
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': tok}, 'finish_reason': None}]})}\n\n"

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
