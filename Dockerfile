FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip python3-venv git ffmpeg libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir \
    transformers accelerate bitsandbytes peft safetensors sentence-transformers \
    gradio fastapi uvicorn ddgs \
    edge-tts sounddevice miniaudio \
    numpy scikit-learn

EXPOSE 8000 7860

CMD ["python3", "adam_chat.py"]
