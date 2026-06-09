# Project Adam

**A self-learning conversational AI that adapts permanently to each user — powered by the COGNET architecture.**

Built on a 4-bit Qwen2.5-3B-Instruct model with LoRA fine-tuning, running entirely on consumer hardware (NVIDIA GTX 1050, 4GB VRAM).

## Features

- **Per-user adaptation** — Detects users, creates profiles, trains LoRA adapters per user
- **Persistent memory** — Episodic (experiences), Semantic (knowledge graph), Neural (gradient-updated attention) stored in SQLite
- **Metacognitive controller** — Automatically decides when to search, clarify, explore, replay, or proceed
- **Streaming output** — Tokens appear one-by-one, no silent wait
- **Web UI** — Gradio interface at `localhost:7860` (`--web`)
- **Voice mode** — Speech-to-speech conversation (`--voice`)
- **REST API** — FastAPI server at `localhost:8000`
- **Web search** — DuckDuckGo → Wikipedia fallback with cache
- **Reward-driven learning** — SFL Q-learning, reward-weighted LoRA curriculum, sentiment analysis

## Quick Start

```bash
pip install -r requirements.txt

# CLI chat
python3 adam_chat.py

# Web UI
python3 adam_chat.py --web

# Voice mode (needs mic + speakers)
python3 adam_chat.py --voice

# REST API server
python3 api_server.py
```

## Architecture

![COGNET Architecture](docs/architecture.md)

```
Metacognitive Controller ─┬─ Persona (adaptive behavioral rules)
                           ├─ Sensory Encoder (efficient coding VAE)
                           ├─ Working Memory (8-turn gated buffer)
                           ├─ Episodic Memory (vector store + reward)
                           ├─ Semantic Memory (schema graph)
                           ├─ Neural Memory (gradient-updated attention)
                           ├─ SFL Module (social feature Q-learning)
                           ├─ User Profiles (per-user state)
                           ├─ Web Search (external knowledge)
                           ├─ Action Selector (fast/slow dual-system)
                           └─ Offline Consolidator (background replay)
```

**Model**: Qwen2.5-3B at 4-bit NF4 (~2.1GB VRAM). Falls back to 1.5B → 0.5B if unavailable.

## Documentation

| File | Contents |
|------|----------|
| `docs/architecture.md` | Project Adam implementation architecture |
| `docs/cognet-architecture.md` | Original COGNET theoretical design (2025-2026 research synthesis) |
| `docs/memory.md` | Memory systems (episodic, semantic, neural) |
| `docs/training.md` | LoRA fine-tuning, curriculum learning, SFL |
| `docs/api.md` | REST API endpoints + usage examples |
| `docs/setup.md` | Installation, requirements, CUDA setup |
| `docs/usage.md` | CLI, Web UI, Voice mode walkthrough |
| `docs/faq.md` | Troubleshooting and common questions |

## Requirements

- Python 3.12+
- CUDA 12.4, NVIDIA GPU with 4GB+ VRAM (GTX 1050 minimum)
- 6GB disk for Qwen2.5-3B model (~2GB quantized)

## License

MIT
