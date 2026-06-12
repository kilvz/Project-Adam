FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update -qq && apt-get install -y -qq \
    python3.12 python3.12-pip python3.12-venv git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r requirements.txt

COPY . .

RUN cp config.yaml.example config.yaml && mkdir -p models personas agent_memory

EXPOSE 8765 7860

ENV PYTHONPATH=/app/src

CMD ["python3", "adam_chat.py"]
