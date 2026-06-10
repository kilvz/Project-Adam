# Project Adam

**A self-learning conversational AI that adapts permanently to each user — powered by the COGNET architecture.**

Built on a 4-bit Qwen2.5-3B-Instruct model with LoRA fine-tuning, running entirely on consumer hardware (NVIDIA GTX 1050, 4GB VRAM). Optionally uses the external public API endpoint (`External` model) for generation while continuing to train the local model via online distillation.

## Features

- **Per-user adaptation** — Detects users, creates profiles, trains LoRA adapters per user
- **Full COGNET architecture** — 14 components verified against `architecture.md`: SensoryEncoder, WorkingMemory (64-slot), EpisodicMemory, SemanticMemory, ProceduralMemory, SpatialMemory, TDCore (with actor network), SFLModule, WorldModel (Bayesian + causal graph), MetacognitiveController (learned policy), LanguageInterface, ActionSelector, OfflineConsolidator
- **Persistent memory** — Episodic (state, action, reward, context), Semantic (schema graph with assimilation/accommodation), Procedural (RL-learned skills with chunking), Spatial (conflict detection, inverse relations)
- **Dual-backend generation** — Local Qwen model (default) or remote API (`external.ai/zen/v1/chat/completions` with `External`). Fallback to local on API failure.
- **Online distillation** — Remote API responses train the local LoRA model every 10 interactions via reward × RPE weighting
- **Metacognitive controller** — Learned MLP policy (5→16→5) with REINFORCE, confidence estimation, learning progress monitoring, 5 canonical actions: EXPLORE, REPLAY, ASK_FOR_HELP, STOP_AND_THINK, SWITCH_STRATEGY
- **Speaker model** — P(language|model) computed via LLM perplexity, language treated as probabilistic Bayesian evidence
- **Streaming output** — Tokens appear one-by-one, no silent wait
- **Web UI** — Gradio interface at `localhost:7860` (`--web`)
- **Voice mode** — Speech-to-speech conversation (`--voice`)
- **REST API** — FastAPI server at `localhost:8000` with OpenAI-compatible `/v1/chat/completions` endpoint
- **OpenAI-compatible API** — `POST /v1/chat/completions` (streaming + non-streaming), `GET /v1/models`. Set `LOCAL_ENDPOINT=http://localhost:8000/v1` in external to use `adam-cognet` model.
- **Web search** — DuckDuckGo → Wikipedia fallback with cache
- **Reward-driven learning** — SFL Q-learning (7 social features), reward-weighted LoRA curriculum, RPE broadcast to all subsystems
- **Configurable via YAML** — `config.yaml` overrides device, model, quantization, generation params, backend mode
- **Structured logging** — Standard `logging` framework with timestamps, levels, and module names

## Quick Start

```bash
pip install -r requirements.txt

# CLI chat
python3 -m project_adam

# Web UI
python3 -m project_adam --web

# Voice mode (needs mic + speakers)
python3 -m project_adam --voice

# REST API server
uvicorn project_adam.api:app --host 0.0.0.0 --port 8000

# Test OpenAI-compatible endpoint
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"adam-cognet","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

### Use with external

```bash
export LOCAL_ENDPOINT="http://localhost:8000/v1"
# external detects adam-cognet model automatically
```

### Use remote API backend

Edit `config.yaml`:
```yaml
backend:
  mode: "api"  # instead of "local"
  api:
    endpoint: "https://external.ai/zen/v1/chat/completions"
    key: ""                    # public endpoint, no key needed
    model: "External"
```

## Architecture

![COGNET Architecture](architecture.md)

```
Metacognitive Controller ─┬─ Sensory Encoder (β-VAE, top-10% sparse, learned prior)
                           ├─ Working Memory (64-slot, token window 2048, attention gating, temporal decay)
                           ├─ Episodic Memory ((s,a,r,c) tuples, symbolic index, temporal compression)
                           ├─ Semantic Memory (schema graph, slot values, prediction error, assimilation/accommodation)
                           ├─ Procedural Memory (RL via RPE, chunking, Q-values)
                           ├─ Spatial Memory (17 relations, conflict detection, inverse inference, traversal)
                           ├─ SFL Module (7 social features, Rescorla-Wagner Q-learning)
                           ├─ User Profiles (per-user state, custom rules)
                           ├─ World Model (Bayesian conjugate priors, causal graph, transition dynamics, speaker model)
                           ├─ Web Search (DDGS → Wikipedia, cached)
                           ├─ Action Selector (dual-system: learned Q-values + trajectory simulation)
                           └─ Offline Consolidator (full 6-step cycle: replay→prioritize→abstract→prune→WM→procedural)
```

**Local model**: Qwen2.5-3B at 4-bit NF4 (~2.1GB VRAM). Falls back to 1.5B → 0.5B if unavailable.

**Remote API**: `external.ai/zen/v1/chat/completions` with `External`. No API key required. Falls back to local model on failure.

**Training**: LoRA adapters (`r=8, target_modules=["q_proj","v_proj"]`) trained every 10 interactions. Data sourced from both local and API responses. Adapters saved per user at `agent_memory/adapters/{user}/`.

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
  mode: "local"                    # "local" or "api"
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

## Documentation

| File | Contents |
|------|----------|
| `architecture.md` | COGNET theoretical architecture (294 lines) |
| `remotearchitecture.md` | Remote API integration layer for COGNET |
| `docs/plan.md` | Architecture alignment plan (phases 1-5) |
| `docs/setup.md` | Installation, requirements, CUDA setup, config, logging |
| `docs/usage.md` | CLI, Web UI, Voice mode walkthrough |
| `docs/api.md` | REST API endpoints + usage examples |
| `docs/faq.md` | Troubleshooting and common questions |

## Requirements

- Python 3.12+
- CUDA 12.4, NVIDIA GPU with 4GB+ VRAM (GTX 1050 minimum)
- 6GB disk for Qwen2.5-3B model (~2GB quantized)
- No API key needed for external remote endpoint

## License

**AGPL v3** with commercial option — see [LICENSE](LICENSE).

Free to use, modify, and distribute under the AGPL v3 terms.
If you use this software in a proprietary product without releasing
your source code, you must purchase a commercial license.
