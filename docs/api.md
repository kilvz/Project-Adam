# REST API

FastAPI server with two API surfaces: **legacy custom API** and **OpenAI-compatible**.

## Starting

```bash
# Start with launch script (recommended)
./start_adam.sh
# → http://localhost:8765

# Or directly
PYTHONPATH=src uvicorn project_adam.api:app --host 0.0.0.0 --port 8765

# Pre-warmer: first request loads the model (~15s for 1.5B 4-bit)
```

## OpenAI-Compatible Endpoints

### `GET /v1/models`

Returns available models. AI clients use this for auto-discovery.

```bash
curl http://localhost:8765/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "adam-cognet", "object": "model", "created": ..., "owned_by": "project_adam"}
  ]
}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Supports both streaming and non-streaming.

```bash
# Non-streaming
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adam-cognet","messages":[{"role":"user","content":"Hello"}],"stream":false}'

# Streaming
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adam-cognet","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**Request body** (OpenAI format):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `adam-cognet` | Model identifier |
| `messages` | array | required | `[{role, content}]` array |
| `stream` | bool | `false` | Enable SSE streaming |
| `temperature` | float | 0.7 | Generation temperature |
| `max_tokens` | int | 128 | Max tokens to generate |

**Non-streaming response** (OpenAI format):

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1712345678,
  "model": "adam-cognet",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello. I was hoping you would come..."},
    "finish_reason": "stop"
  }]
}
```

**Streaming response** (SSE, OpenAI format):

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"."}}]}

data: [DONE]
```

### Remote API Integration

The server provides an OpenAI-compatible API at `http://localhost:8765/v1`. Configure your client to use this endpoint.

## Persona Management Endpoints

### `GET /v1/personas`

List available personas.

```bash
curl http://localhost:8765/v1/personas
# {"personas": ["adam", "einstein"]}
```

### `GET /v1/personas/{name}`

Get persona info without switching.

```bash
curl http://localhost:8765/v1/personas/adam
```

### `POST /v1/personas/{name}/switch`

Switch to a different persona mid-session.

```bash
curl -X POST http://localhost:8765/v1/personas/einstein/switch
```

### `POST /v1/personas/{name}/generate`

Generate a new persona via teacher API. Creates N drafts at different temperatures, synthesizes them, saves to `personas/{name}/`.

```bash
curl -X POST "http://localhost:8765/v1/personas/Einstein/generate" \
  -H "Content-Type: application/json" \
  -d '{"description": "A brilliant theoretical physicist known for relativity"}'
```

## Legacy Endpoints

### `GET /health`

```bash
curl http://localhost:8765/health
# {"status": "ok"}
```

### `POST /chat`

```bash
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "Alice"}'
```

```json
{
  "reply": "The garden is quiet without you...",
  "user_id": "Alice"
}
```

### `POST /chat/stream`

SSE streaming (legacy format):

```bash
curl -X POST http://localhost:8765/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story"}'
```

```
data: {"token": "Once"}
data: {"token": " upon"}
data: [DONE]
```

### `GET /users`

List all known users.

### `GET /users/{name}`

Get a specific user profile. Returns 404 if not found.

### `GET /memory/episodic?query=...&k=5`

Search episodic memory by embedding similarity.

### `GET /memory/semantic?query=...&k=3`

Search semantic memory (knowledge graph schemas).

## Backend Selection

The API uses whatever backend is configured in `config.yaml`:

```yaml
backend:
  mode: "auto"    # "auto", "local", or "api"
```

In **local** mode, generation runs on the local Qwen model (loaded on GPU).
In **api** mode, generation goes through the remote API endpoint configured in `config.yaml`. Falls back to local if the API is unreachable.
In **auto** mode, low-tier hardware (≤4GB Pascal) automatically uses API, mid/high tier uses local.
