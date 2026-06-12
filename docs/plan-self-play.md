# Self-Play Mode — Architecture-Compliant Implementation Plan

## Overview

Self-play lets Adam autonomously generate queries, get responses from the teacher API, and learn from them — all without human interaction. A background thread runs continuously, rotating through query strategies derived from Adam's own knowledge gaps.

**Architecture compliance rule**: The thread generates data only. All learning happens through the existing architecture pipeline: `EpisodicMemory` → `OfflineConsolidator.merge_episodes()` (6-step cycle with RPE prioritization) → `_lora_train_step()` (reads from episodic memory). No `train_from_examples()`, no supervised bypass, no parallel training path.

---

## 1. New File: `src/project_adam/self_play.py` (~170 lines)

### Class: `SelfPlayLearner`

```python
class SelfPlayLearner:
    def __init__(self, agent, config):
        self.agent = agent
        self._running = threading.Event()
        self._thread = None

        self.interval = config.get("interval_seconds", 120)
        self.batch_size = config.get("batch_size", 8)
        self.reward = config.get("reward", 0.85)
        self.strategies = config.get("strategies",
            ["schema", "world_model", "procedural", "creative"])
        self.checkpoint_interval = config.get("checkpoint_interval", 50)

        # Dedup state — deque(maxlen=200) of recent query strings
        self._query_history = deque(maxlen=200)
        self._checkpoint_path = get_memory_dir() / "self_play_checkpoint.json"
        self._load_checkpoint()

        # Stats — dict that agent.stats() can include
        self.stats = {
            "total_queries": 0,
            "total_trained": 0,
            "current_strategy": None,
            "running": False,
            "started_at": None,
            "last_error": None,
        }
```

### Lifecycle

| Method | Purpose |
|---|---|
| `start()` | Set event flag, spawn daemon thread, set `stats["running"] = True` |
| `stop()` | Clear event flag, join thread (timeout=5), save checkpoint, set `running = False` |
| `_loop()` | Main loop (see below) |
| `_call_teacher(query) → str` | Calls `agent.language._api_generate([{"role":"user","content":query}])`. Falls back to `_local_generate()` if API fails. Both methods exist at `language.py:51` and `language.py:95`. |
| `_dedup(queries) → list` | Embeds each query via `agent.episodic_memory.encode()`, checks cosine similarity against `_query_history`. Skips if `>0.85` match to any recent entry. |
| `_log_stats(strategy, count)` | Updates `stats["total_queries"]`, `stats["current_strategy"]` |
| `_save_checkpoint()` | JSON dump of `_query_history` + `stats` to `checkpoint_path` |
| `_load_checkpoint()` | JSON load if exists, else empty |

### `_loop()` — the daemon thread body

```
while self._running.is_set():
    for strategy in self.strategies:
        self.stats["current_strategy"] = strategy
        n = self.batch_size // len(self.strategies)

        queries = self._generate_queries(strategy, n)
        queries = self._dedup(queries)

        for q in queries:
            resp = self._call_teacher(q)
            if not resp:
                continue

            # ── ARCHITECTURE PATH: store in episodic memory ──────────
            # This is the ONLY operation that touches agent state.
            # The episode flows through the existing pipeline:
            #   merge_episodes() → RPE → _lora_train_step()
            self.agent.episodic_memory.add(
                text=q,
                reward=self.reward,
                action=resp,
                context="self_play",
            )
            self._query_history.append(q)
            self.stats["total_queries"] += 1
            self.stats["total_trained"] += 1
            # ──────────────────────────────────────────────────────────

        if self.stats["total_queries"] % self.checkpoint_interval == 0:
            self._save_checkpoint()

    time.sleep(self.interval)
```

The thread never calls `merge_episodes()`, `_lora_train_step()`, or any training function. The metacog's existing REPLAY action triggers consolidation when appropriate.

### Query Generation Strategies — 4 strategies

#### Strategy 1: `_queries_from_schemas(n)`
**Source**: `agent.semantic_memory.schemas` (a `dict[sid, schema]` at `semantic.py:34`)

Each schema has:
- `category` — e.g. `"python"`, `"user_preference"`
- `prediction_error` — float, `1.0` = completely new, `0.0` = perfectly predicted
- `observed_count` — int, how many times this schema was observed
- `facts` — list of strings

**Selection**: Filter schemas where `prediction_error > 0.2` or `observed_count < 3`. Sort by `prediction_error` descending. Pick top N.

**Query template**: `"Explain {category} to me. Focus on what I don't know yet."`

**If no schemas exist**: Generate a generic query like `"Teach me something interesting about science."`

#### Strategy 2: `_queries_from_world_model(n)`
**Source**: `agent.world_model.entities` (a `dict[entity_name, dict[attribute, (mean, var, count)]]` at `world_model.py:9`)

Each entity has attributes with Bayesian posterior `(mean, var, count)`. Higher `var` = higher uncertainty.

**Selection**: For each entity, compute mean `uncertainty()` across all attributes via `self.agent.world_model.uncertainty(entity, attr)` (method at `world_model.py:95`). Sort entities by mean uncertainty descending. Pick top N.

**Query template**: `"What is {entity}? I want to understand it better."`

**If no entities exist**: Fall back to creative strategy.

#### Strategy 3: `_queries_from_procedural_gaps(n)`
**Source**: `agent.procedural_memory.skills` (a `dict[skill_id, Skill]` at `procedural.py:77`)

Each `Skill` object has:
- `.q_value` — float 0-1, learned quality of the skill
- `.success_rate` — float 0-1, `success_count / total_count`
- `.keywords` — `set[str]` of context words from training
- `.action` — str, the stored action/response

**Selection**: Filter skills where `q_value < 0.3` or `success_rate < 0.5`. Sort by `q_value` ascending. Pick top N.

**Query template**: `"How do I handle {keywords}? I've struggled with this."`

**If no skills exist**: Fall back to creative strategy.

#### Strategy 4: `_queries_creative(n)`
**Source**: Teacher API itself

Ask the teacher to suggest a topic. The teacher's reply becomes both the training context and target.

**Query to teacher**: `"Suggest a topic for me to learn about. Give me one specific concept I should explore."`

Process the response: use the response *as* the query text, then ask the teacher to elaborate. Or simpler: use the response directly as the "query" and store it with the teacher's own response as the "action".

**Alternative approach** (simpler): Generate a query from a fixed list or by prompting the local model with `"Generate a question about something you want to learn."`

---

### Dedup Implementation

```python
def _dedup(self, queries):
    if not self._query_history or not self.agent.episodic_memory.embedder:
        return queries  # no embedder = skip dedup
    result = []
    for q in queries:
        q_emb = self.agent.episodic_memory.encode(q)
        is_dup = False
        for recent in self._query_history:
            r_emb = self.agent.episodic_memory.encode(recent)
            sim = float(q_emb @ r_emb / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb) + 1e-8))
            if sim > 0.85:
                is_dup = True
                break
        if not is_dup:
            result.append(q)
    return result
```

---

## 2. Changes to `agent.py` (~20 lines)

### In `__init__()` — add after all components initialized (~line 129)

```python
from .self_play import SelfPlayLearner
self.self_play = None
if SELF_PLAY_CONFIG.get("enabled"):
    self.self_play = SelfPlayLearner(self, SELF_PLAY_CONFIG)
    self.self_play.start()
    logger.info("Self-play started: interval=%ds batch=%d strategies=%s",
                self.self_play.interval, self.self_play.batch_size,
                self.self_play.strategies)
```

### New method: `toggle_self_play(action: str) -> dict`

```python
def toggle_self_play(self, action="status"):
    """Control the self-play background thread.

    Args:
        action: "start", "stop", "restart", or "status" (default)

    Returns:
        dict with loop state and training stats.
    """
    if self.self_play is None:
        if action == "start":
            from .config import SELF_PLAY_CONFIG
            from .self_play import SelfPlayLearner
            self.self_play = SelfPlayLearner(self, SELF_PLAY_CONFIG)
            self.self_play.start()
            return {"status": "started", "stats": self.self_play.stats}
        return {"status": "disabled", "stats": {}}

    if action == "start":
        self.self_play.start()
    elif action == "stop":
        self.self_play.stop()
    elif action == "restart":
        self.self_play.stop()
        self.self_play.start()

    return {"status": "running" if self.self_play.stats["running"] else "stopped",
            "stats": self.self_play.stats}
```

### In `chat()` — optional trigger boost (line 273-274)

The existing metacog-driven REPLAY at `agent.py:273-274` already consolidates all episodes (human + self-play). This needs no change. But we add a lightweight hook: if the metacog selects `EXPLORE` and self-play is running, the agent nudges the thread to generate more aggressively (skip the sleep interval on the next iteration). Implemented via a simple flag on `SelfPlayLearner`:

```python
# In agent.py chat(), inside the EXPLORE branch:
if meta_action == "EXPLORE" and self.self_play is not None:
    self.self_play._trigger_now = True  # next loop iteration runs immediately
```

---

## 3. Changes to `config.py` (~15 lines)

### New module-level block (before `_detect_hardware()`)

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

### In `load_config()`, add parsing after backend config block (after `BACKEND_CONFIG["mode"] = mode`)

```python
sp = cfg.get("self_play", {})
if sp:
    SELF_PLAY_CONFIG.update({
        "enabled": sp.get("enabled", SELF_PLAY_CONFIG["enabled"]),
        "interval_seconds": sp.get("interval_seconds",
                                    SELF_PLAY_CONFIG["interval_seconds"]),
        "batch_size": sp.get("batch_size", SELF_PLAY_CONFIG["batch_size"]),
        "strategies": sp.get("strategies", SELF_PLAY_CONFIG["strategies"]),
        "max_recent_queries": sp.get("max_recent_queries",
                                      SELF_PLAY_CONFIG["max_recent_queries"]),
        "reward": sp.get("reward", SELF_PLAY_CONFIG["reward"]),
        "checkpoint_interval": sp.get("checkpoint_interval",
                                       SELF_PLAY_CONFIG["checkpoint_interval"]),
    })
```

---

## 4. Changes to `config.yaml` (~8 lines)

```yaml
self_play:
  enabled: true                    # auto-start on agent init
  interval_seconds: 120            # seconds between query batches
  batch_size: 8                    # queries per batch
  strategies:
    - schema
    - world_model
    - procedural
    - creative
  reward: 0.85                     # default reward for teacher pairs
```

---

## 5. Changes to `api.py` (~15 lines) — optional manual control

Add endpoint to the existing FastAPI server (`api.py`), not a new MCP server:

```python
@app.post("/v1/self_play")
async def control_self_play(action: str = "status"):
    """Control Adam's autonomous self-play learning loop.

    Actions: "start", "stop", "restart", "status"
    """
    agent = get_agent()
    return agent.toggle_self_play(action)
```

---

## Architecture Compliance Verification

| Architecture requirement | Location in architecture.md | How this plan satisfies it |
|---|---|---|
| **RPE drives all learning** | Section 4a: "δ = R + γ·V(s') - V(s) — RPE drives all learning" | Self-play episodes stored with `reward=0.85`. `OfflineConsolidator._td_replay_prioritized()` computes RPE from them (`consolidator.py:46-84`). The same `_lora_train_step()` trains on them (`agent.py:370-406`). |
| **Metacog controls when to explore/replay** | Section 5: "Strategy selection — when to explore, when to ask for help" | The daemon thread is data generation only. It never calls `merge_episodes()`. The metacog's existing REPLAY action at `agent.py:273-274` triggers consolidation on all episodes (human + self-play). EXPLORE action at `agent.py:266` nudges the thread. |
| **Consolidation is the only "sleep" mechanism** | Section 7: "1. Replay → 2. Prioritize → 3. Abstract → 4. Prune → 5. Update world model → 6. Update procedural policies" | Self-play doesn't have its own training loop. `merge_episodes()` runs the full 6-step cycle, processing self-play episodes identically to human ones. Verified: `consolidator.py:300-350`. |
| **Language as evidence for world model** | Section 4d: "P(model\|experience, language) ∝ P(experience\|model) · P(language\|model) · P(model)" | Teacher responses stored as episodes. During consolidation, `_update_world_model()` at `consolidator.py:211-232` feeds them into `world_model.observe_from_text()`. |
| **Episodic memory stores (state, action, reward, context) tuples** | Section 3a | Self-play episodes are stored via `episodic_memory.add(text=q, reward=0.85, action=resp, context="self_play")`. Matches the existing signature at `episodic.py:50-51`. |
| **Dual-system action selection** | Section 7: "Model-Free + Model-Based" | Self-play bypassed entirely — it generates data, doesn't select actions. The action selection system is unaffected. |
| **No supervised bypass** | Section 1: "minimize I(X;Z) while maximizing reward" | No `train_from_examples()`. No direct LoRA optimizer calls. The path is strictly: `episodic_memory.add()` → `merge_episodes()` → `_lora_train_step()`. |

---

## What the plan explicitly does NOT do

| Rejected approach | Why |
|---|---|
| `train_from_examples(pairs, reward)` | Would bypass episodic → consolidation → RPE pipeline. Creates a supervised learning path that violates Section 4a. |
| Persistent `_get_optimizer()` | The code creates a fresh `AdamW` each `_lora_train_step()` at `agent.py:384`. The plan uses this existing behavior. Verified: the fresh optimizer works fine for batched training. |
| MCP server (`mcp_server.py`) | Self-play control goes through the existing FastAPI (`api.py`) or `toggle_self_play()` method. MCP would add a protocol layer with no benefit. |
| Daemon thread calling `merge_episodes()` | Consolidation is metacog-controlled (Section 5). The thread generates data; the metacog decides when to consolidate. |
| `teacher_generate()` on LanguageInterface | Not needed. `LanguageInterface._api_generate(messages)` at `language.py:51` is public — self-play calls it directly with the right message format. |
| Creative strategy asking teacher for topics then re-asking | Simplified: creative strategy uses the teacher's response directly as training data, or uses a fixed prompt list as fallback. |

---

## Files Changed Summary

| File | Lines Added | Type |
|---|---|---|
| `src/project_adam/self_play.py` | ~170 | **New file** — `SelfPlayLearner` class with loop, query generation (4 strategies), dedup, checkpoint |
| `src/project_adam/agent.py` | ~20 | Edit — import + `__init__` auto-start + `toggle_self_play()` + EXPLORE nudge |
| `src/project_adam/config.py` | ~15 | Edit — `SELF_PLAY_CONFIG` block + `load_config()` parsing |
| `src/project_adam/api.py` | ~15 | Edit — `POST /v1/self_play` endpoint for manual control |
| `config.yaml` | ~8 | Edit — `self_play:` section |
| **Total** | **~228** | |

---

## Implementation Order

1. `src/project_adam/self_play.py` — the full `SelfPlayLearner` class
2. `src/project_adam/config.py` — config block + load parsing
3. `config.yaml` — default config (enabled: true)
4. `src/project_adam/agent.py` — import + init + toggle + EXPLORE nudge
5. `src/project_adam/api.py` — `POST /v1/self_play` endpoint
6. Test: verify thread starts, generates queries, stores in episodic memory, doesn't crash
