# Project Adam — Refactoring & Infrastructure Plan

**Goal**: Make the project maintainable, installable, and contributor-friendly.

## P0 — Immediate (blocking others)

### P0.1 — `requirements.txt`
Scan all imports across the project, produce pinned `requirements.txt` + `requirements-dev.txt`.

### P0.2 — Split `adam_chat.py` into `src/project_adam/` package
```
src/project_adam/
├── __init__.py
├── __main__.py              # python -m project_adam
├── config.py                # YAML config loader
├── agent.py                 # CognitiveAgent
├── persona.py               # Persona
├── memory/
│   ├── __init__.py
│   ├── working.py           # WorkingMemory
│   ├── episodic.py          # EpisodicMemory
│   ├── semantic.py          # SemanticMemory
│   ├── neural.py            # NeuralMemory
│   └── store.py             # SQLiteStore
├── profiles.py              # UserProfileManager
├── sfl.py                   # SFLModule
├── metacog.py               # MetacognitiveController
├── encoder.py               # SensoryEncoder
├── search.py                # WebSearch
├── consolidator.py          # OfflineConsolidator
├── selector.py              # ActionSelector
├── ui/
│   ├── __init__.py
│   ├── cli.py               # CLI REPL
│   ├── webui.py             # Gradio UI
│   └── voice.py             # VoiceMode
└── api.py                   # FastAPI app
```

Keep original `adam_chat.py` as compat shim, delete at end.

### P0.3 — GitHub Actions CI
`.github/workflows/ci.yml` — test + lint on push/PR.

## P1 — Important quality & usability

### P1.1 — YAML config
`config.yaml` for model paths, LoRA params, memory dims, etc. Loaded by `config.py`, overridable via env vars.

### P1.2 — Logging framework
Replace `print()` with `logging.getLogger(__name__)`. CLI handler + file rotation.

### P1.3 — Streaming FastAPI endpoint
SSE endpoint `/chat/stream` for token-by-token streaming.

### P1.4 — `/users` + `/memory` API endpoints
Expose CLI commands as REST endpoints.

### P1.5 — `pyproject.toml`
Make pip-installable: `pip install .` → `adam` CLI command.

---

## Execution order
1. requirements.txt + requirements-dev.txt
2. pyproject.toml
3. Package structure + extract modules
4. Config system
5. Logging
6. GitHub Actions CI
7. API expansion (streaming + endpoints)
8. Cleanup (remove adam_chat.py shim)
