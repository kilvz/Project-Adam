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

# Run (first use downloads the model)
python3 adam_chat.py
```

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

## First Run

On first run, the model (`Qwen/Qwen2.5-3B-Instruct`) is downloaded from HuggingFace (~6GB, ~2GB quantized) and cached in `~/.cache/huggingface/`.

Auto-fallback chain if the download fails:
1. Qwen2.5-3B-Instruct (4-bit NF4)
2. Qwen2.5-1.5B-Instruct (4-bit NF4)
3. Qwen2.5-0.5B-Instruct (fp16)

## Optional: Voice Mode

```bash
pip install faster-whisper edge-tts miniaudio sounddevice
```

Voice mode uses `faster-whisper` (tiny model, ~1GB RAM) for ASR and `edge-tts` for TTS.

## Optional: Docker

```bash
docker build -t project-adam .
docker run --gpus all -p 8000:8000 project-adam python3 api_server.py
```

The Docker image is based on `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and includes all dependencies.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| No env vars required | — | Everything is configured at runtime |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | The model auto-falls back to 1.5B; run `nvidia-smi` to check |
| No /dev/kvm | Run `pip install ninja` for sentence-transformers |
| DuckDuckGo search empty | Wikipedia fallback is used automatically |
| Model download fails | Check network; manual fallback chain handles it |
