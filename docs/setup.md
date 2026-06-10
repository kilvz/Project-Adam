# Setup

## Requirements

- **OS**: Linux (tested on Ubuntu 22.04+)
- **GPU**: NVIDIA with 4GB+ VRAM, CUDA 12.4 (works on GTX 1050 Pascal)
- **RAM**: 8GB+
- **Disk**: 6GB+ for model + runtime data
- **Python**: 3.12+

## Installation

```bash
git clone https://github.com/kilvz/Project-Adam.git
cd Project-Adam
pip install -r requirements.txt
pip install -e .
```

## Package Structure

```
Project-Adam/
├── architecture.md              # COGNET architecture specification
├── remotearchitecture.md        # Remote API integration layer
├── config.yaml                  # User-editable config
├── start_adam.sh                # Launch script with auto-GPU-cleanup
├── external.json                # external provider config
├── src/project_adam/
│   ├── __init__.py              # Public API, calls load_config()
│   ├── __main__.py              # Entry point (CLI arg parsing)
│   ├── agent.py                 # CognitiveAgent orchestrator (all 14 components)
│   ├── api.py                   # FastAPI server + OpenAI-compatible endpoint
│   ├── config.py                # Constants, load_config(), BACKEND_CONFIG
│   ├── language.py              # LanguageInterface (dual-backend: local/API)
│   ├── selector.py              # ActionSelector (dual-system: fast/slow)
│   ├── metacog.py               # MetacognitiveController (learned policy)
│   ├── rl_core.py               # TDCore (TD(λ) + ActorNetwork policy head)
│   ├── sfl.py                   # SFLModule (7 social features, Rescorla-Wagner)
│   ├── encoder.py               # SensoryEncoder (β-VAE, learned prior, top-k sparse)
│   ├── world_model.py           # WorldModel (Bayesian conjugate, causal graph)
│   ├── consolidator.py          # OfflineConsolidator (full 6-step cycle)
│   ├── search.py                # WebSearch (DDGS + Wikipedia, cached)
│   ├── persona.py               # Persona system (28KB max)
│   ├── profiles.py              # UserProfileManager
│   ├── memory/
│   │   ├── store.py             # SQLiteStore (WAL, thread-safe)
│   │   ├── working.py           # WorkingMemory (64-slot, attention gating)
│   │   ├── episodic.py          # EpisodicMemory ((s,a,r,c) tuples, symbolic index)
│   │   ├── semantic.py          # SemanticMemory (graph, assimilation/accommodation)
│   │   ├── procedural.py        # ProceduralMemory (RL via RPE, chunking)
│   │   └── spatial.py           # SpatialMemory (17 relations, conflict detection)
│   └── ui/
│       ├── cli.py               # CLI chat interface
│       ├── webui.py             # Gradio Web UI
│       └── voice.py             # Voice mode
├── tests/                       # 132 tests — pytest
└── docs/                        # Documentation
```

## CUDA Setup

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce GTX 1050
```

If CUDA is not available:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Configuration

Edit `config.yaml`:

```yaml
device: cuda
base_model: Qwen/Qwen2.5-1.5B-Instruct
model_chain:
  - Qwen/Qwen2.5-1.5B-Instruct
memory_dir: agent_memory
persona_path: persona-studio/personas/adam.md
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: torch.float16
  bnb_4bit_quant_type: nf4
generation:
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.9
backend:
  mode: "local"                  # "local" or "api"
  api:
    endpoint: "https://external.ai/zen/v1/chat/completions"
    key: ""
    model: "External"
    timeout: 15
```

## Running

```bash
# CLI chat
python3 -m project_adam

# Web UI
python3 -m project_adam --web

# Voice mode
python3 -m project_adam --voice

# API server
./start_adam.sh
# or: PYTHONPATH=src uvicorn project_adam.api:app --host 0.0.0.0 --port 8765

# With external
# export LOCAL_ENDPOINT="http://localhost:8765/v1"
# Then select "Adam (COGNET)" from external's model picker (Ctrl+P)
```

## First Run

On first run, the model is downloaded from HuggingFace and cached in `~/.cache/huggingface/`.

If loading fails, the agent falls back to API-only mode (model=None) and continues working.

## Testing

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src python3 -m pytest tests/    # 132 tests
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Switch to 0.5B model or use `backend.mode: api` |
| CUBLAS_STATUS_ALLOC_FAILED | Run `./start_adam.sh` (clears GPU context) |
| external timeout | Set `"timeout": 120000` in external.json provider options |
| Module not found | Run `pip install -e .` or set `PYTHONPATH=src` |
