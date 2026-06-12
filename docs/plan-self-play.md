# Autonomous Self-Play Mode — Implementation Plan

## Overview

Self-play mode lets Adam generate its own queries, get responses from the teacher API (remote AI), and train on those responses — all without human interaction. A background loop runs continuously, rotating through multiple query strategies to build diverse knowledge.

---

## 1. New File: `src/project_adam/self_play.py`

### Class: `SelfPlayLearner`

Holds a reference to the `CognitiveAgent` instance. Runs a daemon thread that loops: generate queries → teacher API → train LoRA → sleep.

### Query Generation Strategies (4, rotated)

| # | Strategy | Source | Method | Example Query |
|---|----------|--------|--------|-------------|
| 1 | **Schema-based** | `semantic_memory.schemas` | Pick schemas with high `prediction_error` (>0.2) or low observed count | `"Explain {category} to me"` |
| 2 | **World-model uncertainty** | `world_model.entities` | Sort entities by `uncertainty()` descending, pick top N | `"What is {entity}?"` |
| 3 | **Procedural gaps** | `procedural_memory.skills` | Skills with `success_rate < 0.5` or `q_value < 0.3` | `"How do I {skill_keywords}?"` |
| 4 | **Creative** | Teacher API | Ask teacher: `"Suggest a topic for me to learn about."` Then use teacher's reply as the training target | Dynamic — teacher decides |

### Deduplication

- Maintains `deque(maxlen=200)` of recent query strings
- When generating, encodes query via SentenceTransformer and checks cosine similarity against recent queries
- Skips if similarity > 0.85 to any recent entry
- Also skips if the same exact text was trained in the last 50 entries

### Background Loop

```python
while self._running.is_set():
    for strategy in self.active_strategies:
        queries = self._generate_queries(strategy, n=batch_size // len(strategies))
        queries = self._dedup(queries)
        pairs = []
        for q in queries:
            resp = self.agent.language.teacher_generate(q)
            if resp:
                pairs.append({"input": q, "output": resp})
                self._query_history.append(q)
        if pairs:
            result = self.agent.train_from_examples(pairs, reward=self.reward)
            self._log_stats(strategy, result)
            self._save_checkpoint()
        time.sleep(self.interval_seconds)
```

### Thread Safety

- Uses `threading.Event` for start/stop control
- All shared state (episodic memory, LoRA adapter) is accessed through `agent.*` methods which already have their own locks
- Self-play yields when human chat is active (just sleeps through interval — no explicit yield needed since both use the same model sequentially)

### State Tracking

```python
self.stats = {
    "total_queries_generated": 0,
    "total_trained": 0,
    "avg_loss": 0.0,
    "running": False,
    "current_strategy": None,
    "topics_covered": set(),
    "last_error": None,
    "queries_this_session": 0,
    "started_at": None,
}
```

### Checkpointing

Every `N` queries (configurable, default 50), saves a JSON checkpoint with query history and stats so the loop can resume after a restart without repeating recent work.

---

## 2. Changes to `src/project_adam/agent.py`

### Imports

```python
from .self_play import SelfPlayLearner
```

### `__init__` (end of method, after all components initialized)

```python
from .config import SELF_PLAY_CONFIG
self.self_play = None
if SELF_PLAY_CONFIG.get("enabled"):
    self.self_play = SelfPlayLearner(self, config=SELF_PLAY_CONFIG)
    self.self_play.start()
    logger.info("Self-play mode started")
```

### New method: `toggle_self_play(action: str) -> dict`

```python
def toggle_self_play(self, action="status"):
    if self.self_play is None:
        if action == "start":
            from .config import SELF_PLAY_CONFIG
            self.self_play = SelfPlayLearner(self, config=SELF_PLAY_CONFIG)
            self.self_play.start()
            return {"status": "started"}
        return {"status": "no_self_play"}
    
    if action == "start":
        self.self_play.start()
    elif action == "stop":
        self.self_play.stop()
    elif action == "restart":
        self.self_play.stop()
        self.self_play.start()
    
    return self.self_play.get_status()
```

Note: The existing teacher fallback in `chat()` (lines 274-292) stays unchanged — self-play is separate.

---

## 3. Changes to `src/project_adam/config.py`

### New config block

```python
SELF_PLAY_CONFIG = {
    "enabled": False,
    "interval_seconds": 120,
    "batch_size": 8,
    "strategies": ["schema", "world_model", "procedural", "creative"],
    "max_recent_queries": 200,
    "reward": 0.85,
    "checkpoint_interval": 50,
}
```

### In `load_config()`, add parsing of `self_play:` section

```python
sp = cfg.get("self_play", {})
if sp:
    SELF_PLAY_CONFIG.update({
        "enabled": sp.get("enabled", False),
        "interval_seconds": sp.get("interval_seconds", 120),
        "batch_size": sp.get("batch_size", 8),
        "strategies": sp.get("strategies", ["schema", "world_model", "procedural", "creative"]),
        "max_recent_queries": sp.get("max_recent_queries", 200),
        "reward": sp.get("reward", 0.85),
        "checkpoint_interval": sp.get("checkpoint_interval", 50),
    })
```

---

## 4. Changes to `src/project_adam/mcp_server.py`

### New tool: `adam_self_play(action: str) -> dict`

```python
@mcp.tool()
def adam_self_play(
    action: str = "status",
) -> dict:
    """Control Adam's autonomous self-play learning loop.
    
    Actions:
    - "start" — start/resume the self-play loop
    - "stop" — pause the self-play loop
    - "restart" — stop then start
    - "status" — return current stats
    
    Returns dict with loop state and training stats.
    """
    agent = get_agent()
    return agent.toggle_self_play(action)
```

---

## 5. Changes to `config.yaml`

Add at the end:

```yaml
self_play:
  enabled: false                    # auto-start on agent init
  interval_seconds: 120             # seconds between query batches
  batch_size: 8                     # queries per batch
  strategies:                       # query generation sources
    - schema
    - world_model
    - procedural
    - creative
  reward: 0.85                      # default reward for teacher pairs
```

---

## 6. Behavioral Design

### What happens at startup

1. Agent loads normally (model, memories, teacher config)
2. If `config.yaml:self_play.enabled = true` → `SelfPlayLearner` spawns a daemon thread, starts the loop
3. If `false` → nothing starts. Can be triggered via MCP later

### Runtime behavior

- Self-play runs in a **daemon thread** — exits cleanly when the main process exits
- Each iteration: generate queries → dedup → call teacher API → `train_from_examples()` → sleep
- `train_from_examples()` uses the **persistent optimizer** (`_get_optimizer()` at `agent.py:463`) — no AdamW reset between calls
- The teacher API rate-limit is respected via the interval_seconds sleep
- If the teacher API fails (timeout, error), the loop logs a warning and retries next interval — doesn't crash

### Resource usage estimate

| Resource | Impact |
|----------|--------|
| **VRAM** | +0 MB (teacher API doesn't use local GPU, LoRA training already fits in 2.2 GB) |
| **CPU** | Low — batched LLM inference on teacher API (network-bound) |
| **Disk** | ~1 MB/day for checkpoint + adapter saves |
| **Teacher API** | ~4 calls/min at default 120s interval × 8 batch = 0.07 calls/min. Very light. |

### Coexistence with human chat

- Both share the same LoRA adapter — human interactions and self-play training both update the same weights
- No lock contention: the GIL + PyTorch's CUDA synchronization mean operations naturally serialize
- Self-play yields to CUDA if human chat happens simultaneously (the `train_from_examples` call will block on GPU)

---

## 7. Rationale for Design Decisions

| Decision | Why |
|----------|-----|
| **Thread, not process** | Shared VRAM model in same process; no IPC, no memory duplication |
| **Separate class, not agent method** | Clean separation of concerns; easy to disable/test independently |
| **Teacher API for creative queries** | The model can't generate novel topics it doesn't know — the teacher can |
| **Query dedup via embedding** | Exact match is too strict for varied phrasings; semantic dedup catches repeats |
| **Checkpoint every N queries** | Self-play runs unattended for hours; recovery after crash is needed |
| **Daemon thread** | Guarantees clean shutdown when parent process exits (e.g., SIGTERM) |

---

## 8. Implementation Order

1. `src/project_adam/self_play.py` — the full class
2. `src/project_adam/agent.py` — import + init + toggle method
3. `src/project_adam/config.py` — config load
4. `src/project_adam/mcp_server.py` — MCP tool
5. `config.yaml` — default config block

---

## 9. Files Changed Summary

| File | Lines Added | Change Type |
|------|-------------|-------------|
| `src/project_adam/self_play.py` | ~180 | New file |
| `src/project_adam/agent.py` | ~25 | Edit (+ import, + init code, + `toggle_self_play`) |
| `src/project_adam/config.py` | ~20 | Edit (+ config block, + load section) |
| `src/project_adam/mcp_server.py` | ~20 | Edit (+ tool def) |
| `config.yaml` | ~9 | Edit (+ config block) |
| **Total** | **~254** | |
