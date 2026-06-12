#!/usr/bin/env bash
set -e

PORT="${ADAM_PORT:-8765}"
HOST="${ADAM_HOST:-0.0.0.0}"
PID_FILE="/tmp/adam_api.pid"

OLD_PID=""
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 3
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill -9 "$OLD_PID" 2>/dev/null
            sleep 2
        fi
        echo "Old server terminated."
    fi
    rm -f "$PID_FILE"
fi
fuser -k "${PORT}/tcp" 2>/dev/null || true

echo "Detecting hardware..."
python3 -c "
import sys; sys.path.insert(0, 'src')
from project_adam.config import HARDWARE_TIER, GPU_VRAM_GB, GPU_COMPUTE_CAP
print(f'Tier: {HARDWARE_TIER} | VRAM: {GPU_VRAM_GB}GB | CC: {GPU_COMPUTE_CAP[0]}.{GPU_COMPUTE_CAP[1]}')
" 2>/dev/null

echo "Clearing GPU memory..."
GPU_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | cut -d' ' -f1)
python3 -c "
import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Force fresh CUDA context
    _ = torch.zeros(1, device='cuda')
    del _
    torch.cuda.synchronize()
    print('GPU context reset')
else:
    print('No CUDA available')
" 2>/dev/null || true
sleep 2
GPU_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | cut -d' ' -f1)
echo "GPU memory: ${GPU_BEFORE:-?} MiB → ${GPU_AFTER:-?} MiB"

echo "Starting Project Adam API server on $HOST:$PORT ..."
echo "API: http://localhost:$PORT/v1"
echo ""

PYTHONPATH=src nohup uvicorn project_adam.api:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level warning \
    > /tmp/adam_api.log 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo -n "Waiting for server"
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo ""
        HW_TIER=$(python3 -c "import sys; sys.path.insert(0,'src'); from project_adam.config import HARDWARE_TIER, GPU_VRAM_GB; print(f'{HARDWARE_TIER} ({GPU_VRAM_GB}GB)')" 2>/dev/null || echo "unknown")
        echo "=== Project Adam is ready ==="
        echo "Server PID: $PID | API: http://localhost:$PORT | Hardware: $HW_TIER"
        echo "Models: $(curl -s http://localhost:$PORT/v1/models | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null || echo 'adam-cognet')"
        echo ""
        echo "Preloading model (first request loads it — may take ~15s)..."
        curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d '{"model":"adam-cognet","messages":[{"role":"user","content":"ping"}],"stream":false,"max_tokens":1}' \
          -o /dev/null -w "Model loaded in %{time_total}s" 2>/dev/null &
        echo ""
        echo ""
        echo "=== Ready ==="
        echo "API ready — connect your client to http://localhost:$PORT/v1."
        echo "Note: Running $(grep 'base_model' config.yaml | head -1 | cut -d/ -f2) — responses in ~5-15s."
        echo "Logs: tail -f /tmp/adam_api.log"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "Timed out waiting for server. Check logs: tail -f /tmp/adam_api.log"
exit 1
