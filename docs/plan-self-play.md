# Self-Play + MCP — Architecture-Compliant Plan

## Overview

Two capabilities that share the same architecture rule:

> **The thread generates data only. The MCP tools submit data only. All learning happens through the existing pipeline: `EpisodicMemory` → `OfflineConsolidator.merge_episodes()` (6-step cycle with RPE prioritization) → `_lora_train_step()`. No `train_from_examples()`, no supervised bypass, no parallel training path.**

1. **Self-play**: daemon thread generates (query, teacher_response) pairs into episodic memory during idle time. The metacog's REPLAY action consolidates them.
2. **MCP server**: exposes tools for external AIs to query Adam's knowledge (schemas, world model, skills) and submit experiences (episodes, observations, facts, skills) through the same architecture paths.

---

## Part 1: Self-Play (`src/project_adam/self_play.py`)

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

        self._query_history = deque(maxlen=200)
        self._checkpoint_path = get_memory_dir() / "self_play_checkpoint.json"
        self._load_checkpoint()
        self._run_immediately = threading.Event()

        self.stats = {
            "total_queries": 0,
            "total_trained": 0,
            "current_strategy": None,
            "running": False,
            "started_at": None,
            "last_error": None,
        }
```

### Lifecycle methods

| Method | Purpose |
|---|---|
| `start()` | Set event flag, spawn daemon thread, set `stats["running"] = True` |
| `stop()` | Clear event flag, join thread (timeout=5), save checkpoint, set `running = False` |
| `_loop()` | Main loop (see below) |
| `_call_teacher(query) → str` | Calls `agent.teacher_generate(query)` — a public wrapper on agent.py that handles API→local fallback. |
| `_dedup(queries) → list` | Embeds each query via `agent.episodic_memory.encode()`, checks cosine similarity against `_query_history`. Skips if `>0.85` match. |
| `_log_stats(strategy, count)` | Updates `stats["total_queries"]`, `stats["current_strategy"]` |
| `_save_checkpoint()` | JSON dump of `_query_history` + `stats` |
| `_load_checkpoint()` | JSON load if exists, else empty |
| `_run_immediately` | `threading.Event` set by agent when metacog selects EXPLORE — wakes loop to run immediately |

### `_loop()` body

```python
while self._running.is_set():
    # METACOG GATE: only generate when metacog would choose exploration
    # This keeps strategy selection under metacog control (Section 5).
    if self.agent.metacognitive.last_action not in ("EXPLORE", "ASK_FOR_HELP"):
        if not self._run_immediately.is_set():
            time.sleep(self.interval)
            continue
        self._run_immediately.clear()

    for strategy in self.strategies:
        self.stats["current_strategy"] = strategy
        n = self.batch_size // len(self.strategies)

        queries = self._generate_queries(strategy, n)
        queries = self._dedup(queries)

        for q in queries:
            resp = self._call_teacher(q)
            if not resp:
                continue

            # ARCHITECTURE PATH: store in episodic memory
            self.agent.episodic_memory.add(
                text=q,
                reward=self.reward,
                action=resp,
                context="self_play",
            )

            # IMMEDIATELY COMPUTE RPE so consolidation prioritization
            # (Step 2 of the 6-step cycle) can rank this episode.
            baseline_features = [0.0] * 8
            rpe = self.agent.td_core.update(self.reward, baseline_features)
            if self.agent.episodic_memory.episodes:
                self.agent.episodic_memory.episodes[-1]["rpe"] = rpe

            self._query_history.append(q)
            self.stats["total_queries"] += 1
            self.stats["total_trained"] += 1

        if self.stats["total_queries"] % self.checkpoint_interval == 0:
            self._save_checkpoint()

    time.sleep(self.interval)
```

The thread never calls `merge_episodes()`, `_lora_train_step()`, or any training function. The metacog's REPLAY action triggers consolidation. The thread only generates data and computes RPE to enable prioritization.

### Query generation (4 strategies)

#### Strategy 1: `_queries_from_schemas(n)`
**Source**: `agent.semantic_memory.schemas` — dict of `sid → {category, facts[], prediction_error, observed_count}`

Filter: `prediction_error > 0.2` or `observed_count < 3`. Sort descending by prediction_error. Pick top N.

Template: `"Explain {category} to me. Focus on what I don't know yet."`

Fallback if no schemas: `"Teach me something interesting about science."`

#### Strategy 2: `_queries_from_world_model(n)`
**Source**: `agent.world_model.entities` — dict of `entity → {attribute: (mean, var, count)}`

For each entity, compute mean `uncertainty()` across all attributes. Sort descending. Pick top N.

Template: `"What is {entity}? I want to understand it better."`

Fallback if no entities: use creative strategy.

#### Strategy 3: `_queries_from_procedural_gaps(n)`
**Source**: `agent.procedural_memory.skills` — dict of `skill_id → Skill` with `.q_value`, `.success_rate`, `.keywords`

Filter: `q_value < 0.3` or `success_rate < 0.5`. Sort ascending by q_value. Pick top N.

Template: `"How do I handle {keywords}? I've struggled with this."`

Fallback if no skills: use creative strategy.

#### Strategy 4: `_queries_creative(n)` (two-step)

Step 1 — ask the teacher for a topic:

```
Send:  "Suggest a topic for me to learn about. Give me one specific concept."
Return: topic phrase (e.g. "quantum computing")
```

Step 2 — formulate a query from the topic and get a real response:

```
Send:  "Explain {topic} to me. What should I know about it?"
Return: full teacher explanation

Store: text="Explain {topic} to me..."
       action=<full teacher explanation>
       reward=self.reward
```

This guarantees `text ≠ action` — a real (query, response) learning pair.

The topic from Step 1 is added to `_query_history` for dedup so the same topic won't be requested again. If the teacher's Step 1 response is empty or invalid, the creative query is skipped (no episode stored).

### Dedup

```python
def _dedup(self, queries):
    if not self._query_history or not self.agent.episodic_memory.embedder:
        return queries
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

## Part 2: MCP Server (`src/project_adam/mcp_server.py`)

### Shared agent singleton

Add to `src/project_adam/__init__.py`:

```python
_AGENT_CACHE = None

def get_cached_agent():
    global _AGENT_CACHE
    if _AGENT_CACHE is None:
        from .agent import CognitiveAgent
        _AGENT_CACHE = CognitiveAgent()
    return _AGENT_CACHE
```

Update `api.py` to use `from . import get_cached_agent as get_agent` instead of its local singleton.

### MCP Server structure

```python
"""MCP server for querying Adam's knowledge and submitting learning experiences."""

import logging
import math
from mcp.server import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("Project Adam",
    instructions="Query Adam's knowledge and submit experiences through his learning architecture.")

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        from . import get_cached_agent
        _agent = get_cached_agent()
    return _agent
```

### Knowledge Query Tools (read-only)

#### `adam_query_knowledge(topic: str) -> dict`

Search all memory systems for knowledge about a topic.

```python
@mcp.tool(description="Search semantic schemas, world model, and procedural skills for knowledge about a topic.")
def adam_query_knowledge(topic: str) -> dict:
    """Search all memory systems for knowledge about a topic.

    Args:
        topic: The concept or entity to search for.

    Returns:
        Dict with schemas, world model beliefs, skills, and counts.
    """
    agent = _get_agent()
    tl = topic.lower()

    schemas = []
    for sid, s in agent.semantic_memory.schemas.items():
        if tl in s.get("category", "").lower() or any(tl in f.lower() for f in s.get("facts", [])):
            schemas.append({
                "id": sid, "category": s["category"],
                "facts": s["facts"][-5:],
                "prediction_error": round(s.get("prediction_error", 1.0), 3),
                "observed_count": s.get("observed_count", 0),
                "slots": dict(s.get("slots", {})),
            })

    entities = {}
    for entity, attrs in agent.world_model.entities.items():
        if tl in entity:
            entities[entity] = {
                a: {"mean": round(m, 3), "uncertainty": round(math.sqrt(v), 3), "observations": c}
                for a, (m, v, c) in attrs.items()
            }

    skills = []
    for sid, skill in agent.procedural_memory.skills.items():
        if tl in " ".join(skill.keywords).lower():
            skills.append({
                "id": sid,
                "action": skill.action[:200],
                "q_value": round(skill.q_value, 3),
                "success_rate": round(skill.success_rate, 3),
                "usage_count": skill.usage_count,
            })

    return {
        "topic": topic,
        "schemas": schemas, "schema_count": len(schemas),
        "world_entities": entities, "entity_count": len(entities),
        "skills": skills, "skill_count": len(skills),
    }
```

#### `adam_explain_entity(entity_name: str) -> dict`

Detailed view of a specific entity in the Bayesian world model.

```python
@mcp.tool(description="Get Adam's Bayesian posterior beliefs about a specific entity.")
def adam_explain_entity(entity_name: str) -> dict:
    """Get Adam's Bayesian posterior beliefs about a specific entity.

    Args:
        entity_name: The entity to look up (case-insensitive).

    Returns:
        Dict with attribute-level means, uncertainties, observation counts.
    """
    agent = _get_agent()
    ent = agent.world_model.entities.get(entity_name.lower())
    if not ent:
        return {"entity": entity_name, "found": False}
    attributes = {
        a: {"mean": round(m, 3), "uncertainty": round(math.sqrt(v), 3), "observations": c}
        for a, (m, v, c) in ent.items()
    }
    return {
        "entity": entity_name, "found": True,
        "attributes": attributes,
        "total_observations": sum(c for _, _, c in ent.values()),
    }
```

#### `adam_get_status() -> dict`

Stats across all memory systems.

```python
@mcp.tool(description="Return statistics about Adam's memory systems and self-play state.")
def adam_get_status() -> dict:
    """Return statistics about Adam's memory systems and self-play state."""
    agent = _get_agent()
    ep_count = len(agent.episodic_memory.episodes)
    skills = agent.procedural_memory.skills
    status = {
        "memory": {
            "episodic_episodes": ep_count,
            "semantic_schemas": len(agent.semantic_memory.schemas),
            "world_entities": len(agent.world_model.entities),
            "procedural_skills": len(skills),
        },
        "learning": {
            "avg_skill_q": round(sum(s.q_value for s in skills.values()) / max(len(skills), 1), 3),
            "avg_skill_success": round(sum(s.success_rate for s in skills.values()) / max(len(skills), 1), 3),
        },
        "self_play": {},
    }
    if agent.self_play is not None:
        status["self_play"] = dict(agent.self_play.stats)
    else:
        status["self_play"] = {"running": False}
    return status
```

### Teaching Tools (write via architecture paths)

#### `adam_teach(query: str, response: str, reward: float = 0.85) -> dict`

Submit a (query, response) learning pair. Stored in episodic memory — processed by the next consolidation cycle.

```python
@mcp.tool(description="Teach Adam by submitting a (query, response) pair. Stored in episodic memory, processed during consolidation.")
def adam_teach(query: str, response: str, reward: float = 0.85) -> dict:
    """Teach Adam by submitting a (query, response) learning pair.

    Architecture path: EpisodicMemory.add() → merge_episodes() → _lora_train_step().
    Same path as human chat. No bypass.

    Args:
        query: The question or context.
        response: The correct/expert response.
        reward: Quality signal (0.0-1.0, default 0.85).

    Returns:
        Dict confirming storage.
    """
    agent = _get_agent()
    r = max(0.0, min(1.0, reward))
    agent.episodic_memory.add(text=query, reward=r, action=response, context="mcp_teach")
    return {
        "status": "stored", "query_len": len(query), "response_len": len(response),
        "reward": r, "total_episodes": len(agent.episodic_memory.episodes),
    }
```

#### `adam_observe_entity(entity: str, attribute: str, value: float, confidence: float = 1.0) -> dict`

Submit an observation for Adam's Bayesian world model.

```python
@mcp.tool(description="Submit an observation for Adam's Bayesian world model. Updates posterior belief via conjugate Gaussian.")
def adam_observe_entity(entity: str, attribute: str, value: float, confidence: float = 1.0) -> dict:
    """Submit an observation for Adam's Bayesian world model.

    Architecture path: WorldModel.observe() → conjugate Gaussian posterior update.
    Same path as observe_from_text(). No bypass.

    Args:
        entity: Entity name (e.g., "python", "Einstein").
        attribute: Attribute being observed (e.g., "difficulty", "intelligence").
        value: Observed value (numeric).
        confidence: Reliability (0.0-1.0, default 1.0).

    Returns:
        Dict with updated posterior mean, uncertainty, and count.
    """
    agent = _get_agent()
    agent.world_model.observe(entity, attribute, value, confidence=confidence)
    mean, var, count = agent.world_model.entities.get(entity.lower(), {}).get(attribute, (0, 1, 0))
    return {
        "status": "observed", "entity": entity.lower(), "attribute": attribute,
        "posterior_mean": round(mean, 3),
        "posterior_uncertainty": round(math.sqrt(var), 3),
        "observations": count,
    }
```

#### `adam_teach_fact(category: str, fact: str) -> dict`

Submit a fact for semantic memory. Integrated via assimilation/accommodation.

```python
@mcp.tool(description="Submit a fact for semantic memory. Integrated via Piaget assimilation/accommodation.")
def adam_teach_fact(category: str, fact: str) -> dict:
    """Submit a fact for semantic memory.

    Architecture path: SemanticMemory.add() → assimilation/accommodation.
    If similarity ≥ 0.75 to existing schema → assimilate (update slots).
    If < 0.75 → accommodate (new schema with prediction_error=1.0).

    Args:
        category: Knowledge domain (e.g., "science", "user_preference").
        fact: The factual statement (e.g., "Python is dynamically typed").

    Returns:
        Dict with schema ID, prediction error, and observed count.
    """
    agent = _get_agent()
    sid = agent.semantic_memory.add(category, fact)
    schema = agent.semantic_memory.schemas.get(sid, {})
    return {
        "status": "stored", "schema_id": sid, "category": category,
        "prediction_error": round(schema.get("prediction_error", 1.0), 3),
        "observed_count": schema.get("observed_count", 0),
        "facts": schema.get("facts", [])[-3:],
    }
```

#### `adam_teach_skill(context: str, action: str, reward: float = 0.8) -> dict`

Submit a procedural skill example.

```python
@mcp.tool(description="Submit a procedural skill example. Recorded with Q-learning and automatic chunking of repeated patterns.")
def adam_teach_skill(context: str, action: str, reward: float = 0.8) -> dict:
    """Submit a procedural skill example.

    Architecture path: ProceduralMemory.record() → keyword matching → Q-value update.
    Repeated patterns are automatically chunked into macro-actions.

    Args:
        context: The situation or trigger.
        action: The skill/response to learn.
        reward: How successful (0.0-1.0, default 0.8).

    Returns:
        Dict confirming storage with skill counts.
    """
    agent = _get_agent()
    agent.procedural_memory.record(context, action, reward)
    p_stats = agent.procedural_memory.stats()
    return {
        "status": "stored", "context_len": len(context), "action_len": len(action),
        "reward": reward,
        "total_skills": p_stats["num_skills"],
        "total_chunks": p_stats["num_chunks"],
    }
```

### Learning Trigger Tools

#### `adam_consolidate(rpe: float = 1.0) -> dict`

Run the full 6-step consolidation cycle.

```python
@mcp.tool(description="Run the full 6-step consolidation cycle on all accumulated episodes (human + MCP-taught).")
def adam_consolidate(rpe: float = 1.0) -> dict:
    """Run the full 6-step cognitive consolidation cycle.

    Steps:
    1. Replay: sample episodes from memory
    2. Prioritize: high-RPE events first
    3. Abstract: compress repeated patterns into schemata
    4. Prune: remove redundant/noisy memories
    5. Update world model: Bayesian update
    6. Update procedural policies: offline RL

    Same path as the metacog's REPLAY action.

    Args:
        rpe: Reward prediction error weight (default 1.0).

    Returns:
        Dict with pre/post consolidation stats.
    """
    agent = _get_agent()
    if not agent.episodic_memory.episodes:
        return {"status": "no_episodes", "episode_count": 0}
    before = len(agent.episodic_memory.episodes)
    agent.consolidator.merge_episodes(rpe=rpe)
    return {
        "status": "done",
        "episodes_before": before,
        "episodes_after": len(agent.episodic_memory.episodes),
        "skills": agent.procedural_memory.stats(),
        "schemas": len(agent.semantic_memory.schemas),
        "world_entities": len(agent.world_model.entities),
    }
```

#### `adam_self_play(action: str = "status") -> dict`

Control the autonomous self-play loop.

```python
@mcp.tool(description="Start, stop, restart, or check status of the autonomous self-play learning loop.")
def adam_self_play(action: str = "status") -> dict:
    """Control the autonomous self-play background thread.

    The thread generates (query, teacher_response) pairs into episodic memory
    during idle time. The metacog's REPLAY action consolidates them through
    the full 6-step cycle.

    Args:
        action: "start", "stop", "restart", or "status" (default).

    Returns:
        Dict with loop state and training stats.
    """
    agent = _get_agent()
    return agent.toggle_self_play(action)
```

### Entry point

```python
def run_stdio():
    """Run MCP server over stdio for subprocess integration."""
    import asyncio
    asyncio.run(mcp.run_stdio_async())
```

---

## Part 3: Changes to `agent.py`

### In `__init__()` — add after all components initialized (~line 143)

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

### New public wrapper: `teacher_generate(query: str) -> str`

Provides a clean public interface for self-play to call the teacher API, rather than accessing `language._api_generate()` directly.

```python
def teacher_generate(self, query: str) -> str:
    """Generate a teacher response for a raw query (no persona, no metacog).

    Used by the self-play loop to get expert responses for training data.
    Falls back to local model if the API is unreachable.
    """
    messages = [{"role": "user", "content": query}]
    if self.backend == "api":
        reply = self.language._api_generate(messages)
        if not reply:
            reply = self.language._local_generate(messages)
    else:
        reply = self.language._local_generate(messages)
    return reply.strip() if reply else ""
```

### New method: `toggle_self_play(action: str) -> dict`

```python
def toggle_self_play(self, action="status"):
    """Control the self-play background thread.

    Args:
        action: "start", "stop", "restart", or "status" (default).

    Returns:
        Dict with loop state and training stats.
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

### In `chat()` — EXPLORE nudge (line 273-274)

```python
# After the existing REPLAY check:
if meta_action == "REPLAY":
    self.consolidator.merge_episodes(rpe=rpe)
# Add EXPLORE nudge — wakes the self-play loop immediately:
if meta_action == "EXPLORE" and self.self_play is not None:
    self.self_play._run_immediately.set()
```

---

## Part 4: Changes to `config.py`

### New module-level block

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

### In `load_config()`, add parsing after backend config

```python
sp = cfg.get("self_play", {})
if sp:
    SELF_PLAY_CONFIG.update({
        "enabled": sp.get("enabled", SELF_PLAY_CONFIG["enabled"]),
        "interval_seconds": sp.get("interval_seconds", SELF_PLAY_CONFIG["interval_seconds"]),
        "batch_size": sp.get("batch_size", SELF_PLAY_CONFIG["batch_size"]),
        "strategies": sp.get("strategies", SELF_PLAY_CONFIG["strategies"]),
        "max_recent_queries": sp.get("max_recent_queries", SELF_PLAY_CONFIG["max_recent_queries"]),
        "reward": sp.get("reward", SELF_PLAY_CONFIG["reward"]),
        "checkpoint_interval": sp.get("checkpoint_interval", SELF_PLAY_CONFIG["checkpoint_interval"]),
    })
```

---

## Part 5: Changes to `config.yaml`

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

## Part 6: Changes to `__init__.py` and `api.py`

### `__init__.py` — add shared agent cache

```python
_AGENT_CACHE = None

def get_cached_agent():
    global _AGENT_CACHE
    if _AGENT_CACHE is None:
        from .agent import CognitiveAgent
        _AGENT_CACHE = CognitiveAgent()
    return _AGENT_CACHE
```

### `api.py` — use shared agent

Remove local `_agent = None` and local `get_agent()`. Replace with:

```python
from . import get_cached_agent

def get_agent():
    return get_cached_agent()
```

---

## Part 7: Changes to `__main__.py`

```python
if "--mcp" in sys.argv:
    from .mcp_server import run_stdio
    run_stdio()
```

---

## Architecture Compliance Matrix

| Component | Architecture reference | How this plan satisfies it |
|---|---|---|
| **Self-play thread (metacog gate)** | Section 5 — "strategy selection — when to explore" | Loop only generates when `last_action` is EXPLORE or ASK_FOR_HELP. Metacog retains control. |
| **Self-play thread (RPE)** | Section 4a — "δ = R + γ·V(s') - V(s)" + Section 7 Step 2 — "prioritize: high-RPE events first" | RPE computed immediately after each episode is stored, so consolidation prioritization (Step 2) can rank self-play episodes alongside human ones. |
| **Self-play thread (no training)** | Section 7 — "offline consolidation" | Thread never calls training functions. Metacog's REPLAY triggers consolidation. |
| **MCP adam_teach** | Section 3a — "(state, action, reward, context) tuples" | Stores via `episodic_memory.add()`. Same path as `chat()`. |
| **MCP adam_observe_entity** | Section 4d — "Bayesian update: P(model\|evidence)" | Calls `world_model.observe()` — conjugate Gaussian posterior update. |
| **MCP adam_teach_fact** | Section 3b — "assimilation/accommodation" | Calls `semantic_memory.add()` — full Piaget integration via prediction error gating. |
| **MCP adam_teach_skill** | Section 3c — "Learned via RL, supports chunking" | Calls `procedural_memory.record()` with Q-learning. Stats via public `stats()`. |
| **MCP adam_consolidate** | Section 7 — full 6-step cycle | Calls `consolidator.merge_episodes(rpe)` — identical to REPLAY path at agent.py:273. |
| **MCP adam_query_knowledge** | Section 3b/3c/4d — read-only | Reads in-memory data structures via public methods only. No state mutation. |
| **No supervised bypass** | Section 4a — "RPE drives all learning" | No tool accepts arbitrary weights. All training data flows through `EpisodicMemory` → `merge_episodes()` → `_lora_train_step()`. |

---

## Implementation Order

1. `src/project_adam/config.py` — add `SELF_PLAY_CONFIG` block + load parsing
2. `config.yaml` — add `self_play:` section
3. `src/project_adam/self_play.py` — `SelfPlayLearner` class
4. `src/project_adam/agent.py` — import + init + `toggle_self_play()` + EXPLORE nudge
5. `src/project_adam/__init__.py` — add `get_cached_agent()`
6. `src/project_adam/api.py` — use shared agent singleton
7. `src/project_adam/mcp_server.py` — FastMCP with 9 tools
8. `src/project_adam/__main__.py` — add `--mcp` flag
