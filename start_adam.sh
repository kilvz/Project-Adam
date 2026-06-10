#!/usr/bin/env bash
set -e

PORT="${ADAM_PORT:-8000}"
HOST="${ADAM_HOST:-0.0.0.0}"
PID_FILE="/tmp/adam_api.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Adam server is already running (PID $(cat "$PID_FILE")) on port $(lsof -iTCP -sTCP:LISTEN -P -n -p "$(cat "$PID_FILE")" 2>/dev/null | awk '{print $9}' | tail -1 || echo "$PORT")"
    echo "Open external and select the 'adam-cognet' model."
    exit 0
fi

echo "Starting Project Adam API server on $HOST:$PORT ..."
echo "Connect external: export LOCAL_ENDPOINT=http://localhost:$PORT/v1"
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
        echo ""
        echo "=== Project Adam is ready ==="
        echo "Server PID: $PID"
        echo "API: http://localhost:$PORT"
        echo "Models: $(curl -s http://localhost:$PORT/v1/models | python3 -c "import sys,json; print(', '.join(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null || echo 'adam-cognet')"
        echo ""
echo "To use in external:"
echo "  Option 1 — auto (provider already configured in external.json):"
echo "    Just select 'Adam (COGNET)' from the model picker (Ctrl+P)."
echo "  Option 2 — manual:"
echo "    export LOCAL_ENDPOINT=\"http://localhost:$PORT/v1\""
echo ""
        echo "Logs: tail -f /tmp/adam_api.log"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "Timed out waiting for server. Check logs: tail -f /tmp/adam_api.log"
exit 1
