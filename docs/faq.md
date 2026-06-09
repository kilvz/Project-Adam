# FAQ

## Why is generation so slow?

Adam runs on a GTX 1050 with 4GB VRAM — a Pascal-era entry-level GPU. The 3B model at 0.8 tok/s is the best we can do without hardware upgrade. Streaming ensures the first token appears quickly. See [benchmarks](usage.md#generation-speed).

## Can it run on CPU?

No — the model requires CUDA for 4-bit quantization inference. CPU inference would be impractically slow.

## Where is my data stored?

`agent_memory/memory.db` (SQLite). All conversations, profiles, and learned patterns persist between sessions.

## How do I reset everything?

```bash
rm -rf agent_memory/
```

The next run will create fresh storage.

## DuckDuckGo search returns nothing

DuckDuckGo blocks some environments. The system auto-falls back to Wikipedia API. You can also disable search entirely:

```python
agent.web_search = None  # no search fallback
```

## Can I add new users programmatically?

Yes — send any message with a `user_id` to the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "Bob"}'
```

A fresh profile and LoRA adapter are created automatically.

## How does the persona file work?

`persona-studio/personas/adam.md` contains behavioral rules, phrases, and signatures. It's the immutable foundation. All adaptations happen via per-user profile overlays (rules, phrases, preferences) — the base persona is never modified.

## What happens if I change the base model?

The system auto-detects the model family and reloads LoRA adapters. If the new model has different layer names, adapters may need retraining.

## Why 3B and not 7B?

7B-instruct models at 4-bit require ~5GB VRAM + context. The GTX 1050 has 4GB. 3B at 4-bit fits with ~1.9GB headroom. The auto-fallback chain means if 3B fails, 1.5B loads automatically.

## Does it support streaming in the API?

Yes — the FastAPI server uses `TextIteratorStreamer` internally. The `/chat` endpoint returns the complete reply when generation finishes.

## Can I use a different LLM backend?

The code uses HuggingFace transformers with AutoModelForCausalLM. Any instruct-tuned model with a compatible chat template should work. Change `MODEL_NAME` in `adam_chat.py`.
