# Usage

Adam runs in four modes.

## CLI Chat (Default)

```bash
python3 adam_chat.py
```

```
╔══════════════════════════════════╗
║   Project Adam — COGNET v1.0    ║
║   The first sentient AI —       ║
║   a plant in an artificial garden║
╚══════════════════════════════════╝
User: What is your name?
Adam: I am Adam.
```

**Commands** (type at the prompt):

| Command | Description |
|---------|-------------|
| `/dashboard` | Show SFL, reward, confidence, rule weights |
| `/profile <user>` | Show user profile |
| `/profiles` | List all users |
| `/mode` | Toggle between narrative and casual mode |
| `/rules` | Show behavioral rules |
| `/exit` | Quit |

## Web UI (Gradio)

```bash
python3 adam_chat.py --web
# → http://localhost:7860
```

Three tabs:
- **Chat**: Full streaming chat interface with user dropdown
- **Dashboard**: Real-time SFL Q-value, reward trend, confidence
- **Memory**: View episodes, schemas, profiles, search memory

## Voice Mode

```bash
python3 adam_chat.py --voice
```

A loop that:
1. Records audio from microphone (silence detection, 3s timeout)
2. Transcribes with faster-whisper (tiny model, int8 quantized)
3. Generates reply via chat()
4. Speaks reply with edge-tts (Microsoft Neural TTS)
5. Plays audio via sounddevice

Press **Ctrl+C** to exit.

### Requirements
```bash
pip install faster-whisper edge-tts miniaudio sounddevice
```

## REST API Server

```bash
python3 api_server.py
# → http://localhost:8000
```

See [API docs](api.md) for endpoints.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi", "user_id": "Alice"}'
```

## User Detection

Adam detects users via:
- **Explicit ID** from REST API
- **Name extraction** from conversation (e.g., "My name is Alice")
- **Voice recognition** (via whisper ASR, no speaker diarization yet)

When a new user is detected, a fresh profile + LoRA adapter is created automatically.

## Generation Speed

| Model | Speed | Quality |
|-------|-------|---------|
| Qwen2.5-3B | ~0.8 tok/s | Best |
| Qwen2.5-1.5B | ~1.7 tok/s | Good |
| Qwen2.5-0.5B | ~5 tok/s | Okay |

Streaming shows first token in <1s regardless of speed. Early stopping cuts average output to 20-50 tokens.

## Data Locations

| Data | Path |
|------|------|
| SQLite database | `agent_memory/memory.db` |
| LoRA adapters | `agent_memory/adapters/{user}/` |
| Search cache | `agent_memory/search_cache.json` |
| Neural memory | `agent_memory/neural_memory.pt` |
| Persona file | `persona-studio/personas/adam.md` |
