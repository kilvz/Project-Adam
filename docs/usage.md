# Usage

## CLI Chat

```bash
python3 -m project_adam
```

Commands at the prompt:

| Command | Description |
|---------|-------------|
| `/dashboard` | SFL Q, reward, confidence, rule weights |
| `/profile` | Show current user profile |
| `/users` | List all users |
| `/remove <name>` | Remove a user profile |
| `/memory` | Show recent episodes |
| `/schemas` | Show semantic schemas |
| `/search <q>` | Web search |
| `/stats` | Metacog stats |
| `/save` | Save memory to disk |
| `/exit` | Quit |

## Web UI (Gradio)

```bash
python3 -m project_adam --web
# → http://localhost:7860
```

Three tabs: Chat, Dashboard, Memory.

## Voice Mode

```bash
python3 -m project_adam --voice
```

Requires: `pip install faster-whisper edge-tts miniaudio sounddevice`

## API Server

```bash
./start_adam.sh
# → http://localhost:8765
```

See [API docs](api.md) for endpoints.

## Backend Modes

### Local Mode (default)

Uses Qwen2.5 model loaded on your GPU. Slower but runs entirely offline.

```yaml
backend:
  mode: "local"
```

Model size options in `config.yaml`:

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| Qwen2.5-3B 4-bit | ~2.1 GB | ~0.8 tok/s | Best |
| Qwen2.5-1.5B 4-bit | ~0.8 GB | ~5 tok/s | Good |
| Qwen2.5-0.5B fp16 | ~1 GB | ~10 tok/s | Okay |

### API Mode

Routes generation through external's public endpoint (`External` model). ~2-5s response time. Falls back to local on network failure.

```yaml
backend:
  mode: "api"
```

Local model still loads for LoRA training (online distillation).

## Generation Speed

| Backend | First Response | Subsequent |
|---------|---------------|------------|
| Local 1.5B 4-bit | ~15s (cold start) | ~5-15s |
| Local 0.5B fp16 | ~5s | ~2-5s |
| API (External) | ~3s | ~3s |

## Data Locations

| Data | Path |
|------|------|
| SQLite database | `agent_memory/memory.db` |
| LoRA adapters | `agent_memory/adapters/{user}/` |
| Search cache | `agent_memory/search_cache.json` |
| Persona file | `persona-studio/personas/adam.md` |
