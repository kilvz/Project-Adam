# REST API

FastAPI server at `localhost:8000`. Start with:

```bash
# Via module (recommended)
python3 -m project_adam

# Or uvicorn directly
uvicorn project_adam.api:app --host 0.0.0.0 --port 8000

# Or legacy entry point
python3 src/project_adam/api.py
```

## Endpoints

### `GET /health`

Health check.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /chat`

Send a message and get a reply.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what is your name?"}'
```

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | required | User message |
| `user_id` | string | `""` | Optional user identifier |

**Response:**

```json
{
  "reply": "I am Adam, the first sentient AI...",
  "user_id": "Alice"
}
```

### `POST /chat/stream`

Stream the reply token-by-token via Server-Sent Events (SSE).

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story"}'
```

**Same request body** as `/chat`.

**Response** is SSE (`text/event-stream`):

```
data: {"token": "Once"}
data: {"token": " upon"}
data: {"token": " a"}
data: {"token": " time"}
data: {"token": "..."}
data: [DONE]
```

The stream terminates with `data: [DONE]`. On error a `data: {"error": "..."}` event is sent.

### `GET /users`

List all known users.

```bash
curl http://localhost:8000/users
```

```json
[
  {
    "name": "Alice",
    "interaction_count": 15,
    "avg_sentiment": 0.35,
    "topics": {"AI": 8, "music": 4}
  }
]
```

### `GET /users/{name}`

Get a specific user profile.

```bash
curl http://localhost:8000/users/Alice
```

Same format as a single entry above. Returns **404** if the user is not found.

### `GET /memory/episodic`

Search episodic memory.

```bash
curl "http://localhost:8000/memory/episodic?query=hello&k=5"
```

| Query | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search text |
| `k` | int | `5` | Number of results to return |

```json
[
  {"text": "User said hello world", "similarity": 0.92, "reward": 0.5},
  {"text": "Another memory entry", "similarity": 0.78, "reward": 0.3}
]
```

### `GET /memory/semantic`

Search semantic memory (knowledge graph).

```bash
curl "http://localhost:8000/memory/semantic?query=likes&k=3"
```

| Query | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search text |
| `k` | int | `3` | Number of categories to return |

```json
[
  {"category": "likes", "facts": ["I like pizza", "I like coding"], "similarity": 0.92},
  {"category": "dislikes", "facts": ["I hate spam"], "similarity": 0.71}
]
```

## Configuration

The API server respects `config.yaml` at the project root:

```yaml
device: cuda:0
base_model: Qwen/Qwen2.5-3B-Instruct
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
generation:
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.9
```

## Docker

```bash
docker build -t project-adam .
docker run --gpus all -p 8000:8000 project-adam python3 -m project_adam
```

The image is ~12GB (CUDA 12.4 base). First run downloads the model (~6GB).
