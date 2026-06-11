# Project Adam

**A self-learning conversational AI that adapts permanently to each user — powered by the COGNET architecture.**

[![CI](https://github.com/kilvz/Project-Adam/actions/workflows/ci.yml/badge.svg)](https://github.com/kilvz/Project-Adam/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)

> **Status: Fully implemented.** All 14 COGNET components exist in `src/project_adam/`, verified against `architecture.md`, with 136 passing tests. No stubs, no planned sections — the code is the architecture.

Built on a 4-bit Qwen2.5-3B-Instruct model with LoRA fine-tuning, running entirely on consumer hardware (NVIDIA GTX 1050, 4GB VRAM). Optionally uses remote API endpoint for generation while continuing to train the local model via online distillation.

## Features

### 🧠 Cognitive Architecture (all implemented)

| Component | File | What it does |
|-----------|------|-------------|
| **SensoryEncoder** | `encoder.py` | β-VAE with learned prior, top-10% sparsity, RPE-weighted loss. Hardware-tier-aware (low/mid/high). |
| **VisionEncoder** | `encoder.py` | VAE for visual input (2048→128 latent). Enabled only on mid/high hardware. |
| **AudioEncoder** | `encoder.py` | VAE for audio input (1024→64 latent). Enabled only on mid/high hardware. |
| **WorkingMemory** | `memory/working.py` | 64-slot, 2048-token window, attention-gated eviction, temporal decay, goal/hypothesis tracking. |
| **EpisodicMemory** | `memory/episodic.py` | SQLite-backed (s,a,r,c) tuples, SentenceTransformer indexing, temporal compression. |
| **SemanticMemory** | `memory/semantic.py` | Schema graph (slots + edges), prediction-error gated assimilation, cross-user distillation. |
| **ProceduralMemory** | `memory/procedural.py` | RL-learned skills via RPE, keyword-overlap matching, success-rate tracking, chunking. |
| **SpatialMemory** | `memory/spatial.py` | Directed triple store (17 relations, 200 cap), conflict detection, inverse inference, graph traversal. |
| **TDCore** | `rl_core.py` | TD(λ) with linear V(s), ActorNetwork policy head (8→64→5), eligibility traces, RPE broadcast. |
| **SFLModule** | `sfl.py` | 7 social features, Rescorla-Wagner Q-learning, compute_temperature() for action selection. |
| **WorldModel** | `world_model.py` | Bayesian conjugate priors, causal graph, observation from text, transition simulation, speaker model. |
| **MetacognitiveController** | `metacog.py` | Learned MLP policy (5→16→5, REINFORCE), confidence estimation, strategy selection, 5 canonical actions. |
| **LanguageInterface** | `language.py` | Dual-backend generation (local / remote API), persona builder, behavioral rules, utterance-likeness scoring. |
| **ActionSelector** | `selector.py` | Dual-system: pattern-based fast path + world model slow path, learned Q-values, trajectory simulation. |
| **OfflineConsolidator** | `consolidator.py` | Full 6-step cycle (TD replay → prioritize → abstract → prune → world model → procedural). Metacog-triggered. |

### 🚀 User-Facing

- **Per-user adaptation** — User detection, profiles, per-user LoRA adapters at `agent_memory/adapters/{user}/`
- **Dual-backend generation** — Local Qwen (auto-detected hardware tier) or remote API (`external.ai/zen` / OpenAI-compatible)
- **Online distillation** — Remote API responses train the local LoRA model, no separate pipeline needed
- **Hardware auto-detection** — Low tier (≤4GB Pascal) → API mode; Mid (8GB+ Volta) / High (24GB+ Ampere) → local mode
- **Streaming output** — Tokens appear one-by-one
- **Web UI** — Gradio interface at `localhost:7860` (`--web`)
- **Voice mode** — Speech-to-speech conversation (`--voice`)
- **REST API** — FastAPI + OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming), `/v1/models`
- **Web search** — DuckDuckGo for general search, Wikipedia for knowledge (separate methods, independent caches)
- **Reward-driven learning** — SFL Q-learning (7 social features), RPE broadcast to all subsystems, reward-weighted curriculum
- **Configurable via YAML** — `config.yaml` overrides device, model chain, quantization, generation params, backend mode

## Quick Start

```bash
pip install -r requirements-dev.txt

# CLI chat
python3 -m project_adam

# Web UI
python3 -m project_adam --web

# Voice mode (needs mic + speakers)
python3 -m project_adam --voice

# REST API server
uvicorn project_adam.api:app --host 0.0.0.0 --port 8765

# Test OpenAI-compatible endpoint
curl http://localhost:8765/v1/models
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adam-cognet","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### Use with openai compatible chat UI

```bash
export LOCAL_ENDPOINT="http://localhost:8765/v1"

```

### Use remote API backend

Edit `config.yaml`:
```yaml
backend:
  mode: "auto"  # auto-detects: low hardware → API, mid/high → local
  api:
    endpoint: "https://<remotebackend>/v1/chat/completions"
    key: ""                    # endpointkey
    model: "ai-model"
```

## Architecture

![COGNET Architecture](architecture.md)

```
Metacognitive Controller ─┬─ Sensory Encoder (β-VAE, top-10% sparse, learned prior)
                           ├─ Working Memory (64-slot, token window 2048, attention gating, temporal decay)
                           ├─ Episodic Memory ((s,a,r,c) tuples, SentenceTransformer + SQLite)
                           ├─ Semantic Memory (schema graph, slot values, prediction error, assimilation)
                           ├─ Procedural Memory (RL via RPE, chunking, Q-values)
                           ├─ Spatial Memory (17 relations, conflict detection, inverse inference, traversal)
                           ├─ SFL Module (7 features, Rescorla-Wagner Q-learning)
                           ├─ User Profiles (per-user state, custom rules)
                           ├─ World Model (Bayesian conjugate, causal graph, speaker model)
                           ├─ Web Search (DDGS general + Wikipedia knowledge, separate)
                           ├─ Action Selector (dual-system: learned Q-values + trajectory simulation)
                           └─ Offline Consolidator (6-step cycle: replay→prioritize→abstract→prune→WM→procedural)
```

### Hardware Tier Adaptations

| Tier | Criteria | Behavior |
|------|----------|----------|
| **Low** | ≤4GB VRAM, Pascal | `backend.mode=api`, encoder loss `task_weight=0.1`, speaker model uses perplexity proxy, VisionEncoder/AudioEncoder disabled |
| **Mid** | 8GB+ VRAM, Volta+ | `backend.mode=local`, full encoder loss, normalized log-probability speaker model |
| **High** | 24GB+ VRAM, Ampere+ | Same as Mid, plus `flan-t5-large` loaded as encoder-decoder backend |

### Local Model

Qwen2.5-3B at 4-bit NF4 (~2.1GB VRAM). Falls back to 1.5B → 0.5B if unavailable. Tries each model in `model_chain` until one loads.

### Remote API

`<remoteapibackend>/v1/chat/completions` with `ai-model`. API key required. Falls back to local model on failure. In `auto` mode, low-tier hardware automatically uses the API.

### Training

LoRA adapters (`r=8, target_modules=["q_proj","v_proj"]`) trained every 10 interactions. Data sourced from both local and API responses (online distillation). Adapters saved per user at `agent_memory/adapters/{user}/`.

## Configuration

`config.yaml` supports all runtime options:

```yaml
device: cuda                       # "cuda" or "cpu"
base_model: Qwen/Qwen2.5-3B-Instruct
model_chain:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-1.5B-Instruct
  - Qwen/Qwen2.5-0.5B-Instruct

backend:
  mode: "auto"                     # "auto", "local", or "api"
  api:
    endpoint: "https://external.ai/zen/v1/chat/completions"
    key: "${OPENAI_API_KEY}"
    model: "External"
    timeout: 60

generation:
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.9
```

## Testing

```bash
# All tests
PYTHONPATH=src python3 -m pytest tests/ -v

# Single file
PYTHONPATH=src python3 -m pytest tests/test_search.py -v
```

**136 tests** covering all components:
- Encoder (VAE forward, loss, sparsity, hardware tier)
- Memory (working, episodic, semantic, procedural, spatial)
- RL core (TD update, actor policy, eligibility traces)
- SFL (Q-learning, batch, negative reward)
- Metacognitive controller (MLP policy, REINFORCE)
- Language interface (generate, build_prompt, behavioral rules, utterance likeness)
- Action selection (Q-values, trajectory sim, dual system)
- Integration (full chat flow, user detection, reward tracking, SFL updates, episodic memory)
- Web search (DDGS, Wikipedia, cache, independent methods)
- Profiles, persona, API endpoints

## Documentation

| File | Contents |
|------|----------|
| `architecture.md` | COGNET theoretical architecture (294 lines) |
| `docs/setup.md` | Installation, requirements, CUDA setup, config, logging |
| `docs/usage.md` | CLI, Web UI, Voice mode walkthrough |
| `docs/api.md` | REST API endpoints + usage examples |
| `docs/training.md` | LoRA adaptation, RPE, online distillation |
| `docs/memory.md` | All 6 memory systems in detail |
| `docs/faq.md` | Troubleshooting and common questions |
| `docs/wiring-audit.md` | Data flow audit against architecture |

## Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (GTX 1050 minimum) — or use remote API backend
- 6GB disk for Qwen2.5-3B model (~2GB quantized)
- No API key needed for external remote endpoint

## License

**AGPL v3** with commercial option — see [LICENSE](LICENSE).

Free to use, modify, and distribute under the AGPL v3 terms.
If you use this software in a proprietary product without releasing
your source code, you must purchase a commercial license.
