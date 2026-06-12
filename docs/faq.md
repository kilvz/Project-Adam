# FAQ

## General

**Q: Why is Adam so slow locally?**

A: GTX 1050 (Pascal, sm_61) doesn't support efficient 4-bit dequantization kernels. Use `backend.mode: api` in config.yaml for fast responses via a remote API endpoint.

**Q: Why does Adam sound like an assistant?**

A: Previous versions used `"assistant"` role labels. The persona prompt now explicitly states "You are NOT an assistant" and working memory uses `"adam"` as the role. Small models (<3B) may still default to assistant patterns due to RLHF training.

**Q: Can Adam remember me across sessions?**

A: Yes. User profiles with name, topics, sentiment history, custom rules, and LoRA adapters persist in SQLite. The persona prompt includes the user's name every turn.

## Technical

**Q: What does the "backend" config do?**

A: Two modes: `local` runs Qwen on your GPU; `api` routes generation through a remote API endpoint. LoRA training continues regardless.

**Q: What port does the API use?**

A: Default is 8765 (not 8000 — port 8000 is often taken by printer services). Set via `ADAM_PORT=8000 ./start_adam.sh`.

**Q: How do I connect a remote API client?**

A: Point your OpenAI-compatible client to `http://localhost:8765/v1`. The server provides `GET /v1/models` and `POST /v1/chat/completions`.

**Q: Why does the first request take 15 seconds?**

A: Cold start — the model loads lazily on first request. `start_adam.sh` preloads it with a warmup ping so subsequent requests are faster.

## Errors

**Q: "CUDA error: CUBLAS_STATUS_ALLOC_FAILED"**

A: GPU memory is fragmented. Run `./start_adam.sh` — it kills old processes, clears CUDA cache, and forces a fresh GPU context.

**Q: "SQLite objects created in a thread can only be used in that same thread"**

A: Already fixed — `check_same_thread=False` is set on all connections.

**Q: "AttributeError: 'CognitiveAgent' object has no attribute 'model'"**

A: The model failed to load (OOM). The agent now tolerates this — it sets `model=None` and continues in API-only mode. Switch to API backend or a smaller model.

**Q: API shows "Internal Server Error"**

A: Increase `timeout` in config.yaml under `backend.api.timeout`. The first request includes model loading time.

## Architecture

**Q: What components does Adam have?**

A: 20 COGNET components: SensoryEncoder, VisionEncoder, AudioEncoder, WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory, SpatialMemory, DiffMemory, TDCore, SFLModule, WorldModel, WebSearch, MetacogController, LanguageInterface, ActionSelector, OfflineConsolidator, SelfPlayLearner, Persona, UserProfileManager.

**Q: Is neural memory used?**

A: No — `NeuralMemory` was removed. It was not specified in `ARCHITECTURE.md`.

**Q: How does speech/voice work?**

A: Voice mode uses `faster-whisper` (ASR) and `edge-tts` (TTS). No speaker diarization yet.
