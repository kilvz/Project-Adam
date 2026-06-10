# Setup

## Requirements

- **OS**: Linux (tested on Ubuntu 22.04+)
- **GPU**: NVIDIA with 4GB+ VRAM, CUDA 12.4 (works on GTX 1050 Pascal)
- **RAM**: 8GB+
- **Disk**: 6GB+ for model + runtime data
- **Python**: 3.12+

## Installation

```bash
# Clone the repo
git clone https://github.com/kilvz/Project-Adam.git
cd Project-Adam

# Install dependencies
pip install -r requirements.txt

# Optional: install in editable mode (recommended)
pip install -e .

# Run CLI
python3 -m project_adam
```

## Package Structure

```
Project-Adam/
├── src/
│   └── project_adam/
│       ├── __init__.py        # Public API, calls load_config()
│       ├── __main__.py        # Entry point: CLI arg parsing, setup_logging()
│       ├── agent.py           # CognitiveAgent orchestrator
│       ├── api.py             # FastAPI app with REST endpoints
│       ├── config.py          # Constants, load_config(), setup_logging()
│       ├── selector.py        # TextIteratorStreamer, action selection
│       ├── metacog.py         # Metacognitive controller
│       ├── sfl.py             # Social Feature Q-learning
│       ├── search.py          # Web search (DDGS + Wikipedia)
│       ├── consolidator.py    # Offline consolidation
│       ├── encoder.py         # SensoryEncoder (VAE)
│       ├── persona.py         # Persona system
│       ├── profiles.py        # User profiles + LoRA adapters
│       ├── utils.py           # extract_facts, compute_implicit_reward, etc.
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── store.py       # SQLiteStore (WAL, thread-safe)
│       │   ├── working.py     # WorkingMemory (8-turn gated buffer)
│       │   ├── episodic.py    # EpisodicMemory (vector store)
│       │   ├── semantic.py    # SemanticMemory (schema graph)
│       │   └── neural.py      # NeuralMemory (gradient-updated attention)
│       └── ui/
│           ├── __init__.py
│           ├── cli.py         # CLI chat interface
│           ├── webui.py       # Gradio Web UI
│           └── voice.py       # Voice mode
├── tests/                     # 139 tests — pytest
├── config.yaml                # User-editable config overrides
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

Backward compatibility: running `python3 adam_chat.py` (legacy entry point) still works via the compat shim.

## CUDA Setup

Verify CUDA:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce GTX 1050
```

If CUDA is not available:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Configuration

Edit `config.yaml` at the project root to override defaults without modifying code:

```yaml
device: cuda:0
base_model: Qwen/Qwen2.5-3B-Instruct
model_chain:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-1.5B-Instruct
  - Qwen/Qwen2.5-0.5B-Instruct
memory_dir: null           # null = default (agent_memory/)
persona_path: null         # null = default (persona-studio/personas/adam.md)
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
generation:
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.9
```

Settings are merged with code defaults at import time. Omitted keys keep their internal defaults.

## First Run

On first run, the model (`Qwen/Qwen2.5-3B-Instruct`) is downloaded from HuggingFace (~6GB, ~2GB quantized) and cached in `~/.cache/huggingface/`.

Auto-fallback chain if the download fails:
1. Qwen2.5-3B-Instruct (4-bit NF4) — ~2.1GB VRAM, ~0.8 tok/s
2. Qwen2.5-1.5B-Instruct (4-bit NF4) — ~1.2GB VRAM, ~1.7 tok/s
3. Qwen2.5-0.5B-Instruct (fp16) — ~0.6GB VRAM, ~3 tok/s

The first run also creates the `agent_memory/` directory for SQLite databases, adapters, and cache.

## Logging

Logging is configured automatically when running via `python3 -m project_adam`:

```
2026-06-10 10:15:30 [INFO] project_adam.agent: CognitiveAgent initialized with Qwen/Qwen2.5-3B-Instruct
2026-06-10 10:15:31 [INFO] project_adam.config: Memory directory: /path/to/agent_memory
2026-06-10 10:15:32 [INFO] project_adam.profiles: Loaded 3 user profiles
```

Format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`

Log level defaults to `INFO`. Set the `LOG_LEVEL` environment variable to override:
```bash
LOG_LEVEL=DEBUG python3 -m project_adam
```

## Optional: Voice Mode

```bash
pip install faster-whisper edge-tts miniaudio sounddevice
```

Voice mode uses `faster-whisper` (tiny model, ~1GB RAM) for ASR and `edge-tts` for TTS.

## Optional: Docker

```bash
docker build -t project-adam .
docker run --gpus all -p 8000:8000 project-adam python3 -m project_adam
```

The Docker image is based on `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and includes all dependencies.

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
PYTHONPATH=src python3 -m pytest tests/

# Run with coverage
PYTHONPATH=src python3 -m pytest tests/ --cov=project_adam
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | The model auto-falls back to 1.5B; run `nvidia-smi` to check |
| No /dev/kvm | Run `pip install ninja` for sentence-transformers |
| DuckDuckGo search empty | Wikipedia fallback is used automatically |
| Model download fails | Check network; manual fallback chain handles it |
| `module not found: project_adam` | Run `pip install -e .` or set `PYTHONPATH=src` |
