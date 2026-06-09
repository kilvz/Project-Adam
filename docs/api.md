# REST API

FastAPI server at `localhost:8000`. Start with:

```bash
python3 api_server.py
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

## Docker

```bash
docker build -t project-adam .
docker run --gpus all -p 8000:8000 project-adam python3 api_server.py
```

The image is ~12GB (CUDA 12.4 base). First run downloads the model (~6GB).
