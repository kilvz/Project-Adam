# REMOTE — Remote API Integration for COGNET

Extends the **COGNET architecture** (`architecture.md`) with a remote LLM backend while preserving local model training and all existing components.

---

## Core Principle

**Online distillation** — the remote teacher generates responses; the local student learns from them through RL-weighted LoRA fine-tuning. Every interaction trains the local model regardless of which backend served the reply.

```
Remote API (teacher) ──► generates reply ──► stored in episodic memory
                                                      │
Local Model (student) ◄── LoRA train on (input, reply) pairs with reward × RPE weight
```

The remote API is interchangeable: OpenAI, Anthropic, vLLM, Ollama, or any OpenAI-compatible endpoint.

---

## High-Level Architecture

Only `LanguageInterface` changes. Everything else is identical to `architecture.md`:

```
┌─────────────────────────────────────────────────────────────┐
│                    METACOGNITIVE CONTROLLER                  │
│  (unchanged from architecture.md)                           │
└────────────────────┬────────────────────────────────────────┘
                     │ controls
    ┌────────────────┼────────────────────┐
    ▼                ▼                    ▼
┌──────────┐  ┌──────────┐  ┌──────────────────────────────────┐
│ SENSORY  │  │ WORKING  │  │  LANGUAGE INTERFACE              │
│ ENCODERS │◄─┤ MEMORY   │◄─┤  ┌───────────────────────────┐  │
│ (same)   │  │ (same)   │  │  │ backend: "local" | "api"  │  │
└────┬─────┘  └────┬─────┘  │  │                           │  │
     │              │       │  │  local_model.generate()    │  │
     ▼              ▼       │  │  api_model.generate() ───► │  │
┌───────────────────────┐   │  └───────────────────────────┘  │
│  LONG-TERM MEMORY     │   └──────────────────────────────────┘
│  (same)               │
│  ┌──────────┐ ┌──────┐│              │
│  │ Episodic │ │Semant││  ────────────┘ (both paths write to episodic)
│  │ (stores  │ │(same)││
│  │  ALL     │ └──────┘│
│  │  replies)│ ┌──────┐│
│  │          │ │Proced││
│  └──────────┘ └──────┘│
│       ┌───────────────┘│
│       ▼                ▼
│ ┌──────────────────────────┐
│ │  OFFLINE CONSOLIDATOR   │
│ │  (same — replays ALL    │
│ │   episodes regardless    │
│ │   of backend)            │
│ └──────────────────────────┘
└──────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    LEARNING ENGINE                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ RL Core  │ │ Efficient│ │  Social  │ │   Bayesian   │  │
│  │ (same)   │ │ Coding   │ │ Feature  │ │  World Model │  │
│  │          │ │ (same)   │ │ Learning │ │   (same)     │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│                    ACTION SELECTION (same)                   │
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │ Model-Free (fast)  │  │ Model-Based (slow/deliberate)│   │
│  └────────────────────┘  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

Only the **Language Interface** component gains a new backend path. All other components operate identically regardless of backend choice.

---

## Detailed Component Design

### Components 1–5, 7–8: Unchanged

The following components are **identical** to their descriptions in `architecture.md`:

| Component | Architecture Section | Status |
|---|---|---|
| 1. Sensory Encoders | §1 | Unchanged |
| 2. Working Memory | §2 | Unchanged |
| 3a. Episodic Memory | §3a | Unchanged |
| 3b. Semantic Memory | §3b | Unchanged |
| 3c. Procedural Memory | §3c | Unchanged |
| 3d. Spatial Memory | §3d | Unchanged |
| 4a. RL Core | §4a | Unchanged |
| 4b. Efficient Coding Objective | §4b | Unchanged |
| 4c. Social Feature Learning | §4c | Unchanged |
| 4d. Bayesian World Model | §4d | Unchanged |
| 5. Metacognitive Controller | §5 | Unchanged |
| 7. Action Selection | §7 | Unchanged |
| 8. Consolidation Cycle | §8 | Unchanged |

The only change is a **new field** in every episodic entry recording which backend generated the action:

```python
entry = {
    "state": user_input,
    "action": reply,
    "reward": reward,
    "context": wm_context,
    "rpe": rpe,
    "backend": "local" | "api",   # ← new field
    "ts": time.time(),
}
```

This field enables:
- Filtering LoRA training data by backend provenance
- Tracking which backend produces higher-reward responses
- Selective consolidation by backend

---

### 6. Language Interface (Modified)

**Purpose:** Generate natural language responses. Supports two backends.

#### Configuration

```
backend:
  mode: "local" | "api"       # inference backend
  api:
    endpoint: "https://api.openai.com/v1/chat/completions"
    key: "${OPENAI_API_KEY}"   # env var or direct value
    model: "gpt-4"            # remote model name
    timeout: 30               # request timeout in seconds
```

The backend is selectable at startup via `config.yaml`. It can also be overridden per-turn by the metacognitive controller (e.g., use local for routine greetings, API for complex reasoning).

#### Generation

```
Input: messages, meta_action, temperature, token_callback

if backend == "local":
    → self.model.generate(...)    # existing Qwen path (architecture.md §6)
elif backend == "api":
    → self._api_generate(...)     # remote API path

Output: (reply_text, used_search, web_context)
```

**Local path** — identical to `architecture.md` §6. Uses `AutoModelForCausalLM.generate()` with `TextIteratorStreamer` for streaming.

**API path** — calls an OpenAI-compatible chat completions endpoint:

```
POST {endpoint}
Headers:
  Authorization: Bearer {key}
  Content-Type: application/json

Body:
{
  "model": "{model}",
  "messages": [
    {"role": "system", "content": "{system_prompt}"},
    {"role": "user", "content": "{user_input}"}
  ],
  "temperature": {temperature},
  "max_tokens": 128,
  "stream": true
}

Response (SSE stream):
  data: {"choices": [{"delta": {"content": "token"}}]}
```

Streaming uses the same `token_callback` interface as the local path — the CLI/Gradio UI are unaffected.

**Self-talk** — both backends can generate self-talk via the same prompt. No change needed to `generate_self_talk()`.

**Behavioral rules** — `apply_behavioral_rules()` runs on the reply text regardless of backend. No change needed.

#### Speaker Model

The architecture (architecture.md §4d, line 188) requires `P(language|model)` via a speaker model.

| Backend | Implementation |
|---|---|
| **Local** | `compute_utterance_likeness(text)` — passes text through local model, computes perplexity from loss. Confidence = `1 - perplexity/100` |
| **API** | Returns fixed `confidence = 0.5` — perplexity cannot be computed through a remote endpoint. The world model receives language evidence with neutral confidence |

This is the **only functional difference** between backends for the world model. The confidence value feeds into `WorldModel.observe()` as the `confidence` parameter, scaling the observation variance.

#### User Detection

`detect_user()` operates on raw text and does not require a model — unchanged regardless of backend.

---

## Learning & Consolidation Cycle

The cycle is **identical** to `architecture.md` §8, with one addition:

### Online Phase

```
Observe → WM → Act (local OR api) → Get reward → Update
```

The `backend` field is recorded in the episodic entry. All local learning signals (TD error, RPE broadcast, SFL update, encoder train) operate identically.

### Offline Phase (Consolidation)

```
1. Replay: sample episodes from memory (any backend)
2. Prioritize: high-RPE events first
3. Abstract: compress repeated patterns into schemata
4. Prune: remove redundant/noisy memories
5. Update world model: Bayesian update
6. Update procedural policies: offline RL
7. LoRA train: fine-tune local model on high-reward episodes
   └── Filters by backend field, trains on API-generated data too
```

Step 7 is the **online distillation** mechanism:

```
_lora_train_step():
  candidates = episodes with reward > -0.3
              AND (backend == "api" OR backend == "local")  # both
  sort by reward desc
  train local LoRA on top-5 (input → reply) pairs
  weight = reward × RPE_scale
```

The local model learns from:
- Its own high-reward responses (reinforcement)
- The remote API's high-reward responses (imitation / distillation)

Over time, the local model internalizes the remote API's successful patterns. If the API is ever removed, the local model retains what it learned.

---

## Implementation Phases

### Phase 1: Config + LanguageInterface API Path

| File | Change |
|---|---|
| `config.yaml` | Add `backend.mode`, `backend.api.endpoint`, `backend.api.key`, `backend.api.model` |
| `config.py` | Load new config fields; expand `load_config()` to read `backend` section |
| `language.py` | Add `self.backend` field; add `_api_generate()` method using `requests` + SSE parsing |
| `language.py` | `generate()` dispatches to `_local_generate()` or `_api_generate()` based on `self.backend` |
| `language.py` | `compute_utterance_likeness()` returns 0.5 when `self.backend == "api"` |

### Phase 2: Backend Tracking in Episodic Memory

| File | Change |
|---|---|
| `agent.py` | Pass `backend` to `episodic_memory.add()` as a field |
| `memory/episodic.py` | `add()` accepts optional `backend` parameter |

### Phase 3: Selective LoRA Training

| File | Change |
|---|---|
| `agent.py` | `_lora_train_step()` optionally filters episodes by `backend` for targeted distillation |

---

## Key Novel Properties

1. **Backend transparency** — the agent stores which backend generated each response, enabling tracking and comparison
2. **Online distillation** — the local model learns from remote API outputs through the existing LoRA training loop, no separate training pipeline needed
3. **Gradual independence** — as the local model internalizes remote patterns, the API can be phased out for common cases
4. **Unified learning signals** — RPE, reward, and consolidation operate identically regardless of backend
5. **No architectural changes** — only `LanguageInterface` is modified; all other components are unaffected

---

## Tradeoffs

| Feature | Local Only | API Hybrid |
|---|---|---|
| VRAM usage | ~2.1GB (3B 4-bit) | ~0.5GB (0.5B fp16 or no model) |
| Response quality | Qwen 3B | GPT-4 / Claude / any API model |
| Latency | ~1.3s/token | ~0.5–5s total (network dependent) |
| Speaker model | Perplexity-based confidence | Fixed 0.5 confidence |
| LoRA training | Full local fine-tuning | Trains local model on API outputs |
| Offline capability | Full | Requires API for generation |
| Cost | Free (compute only) | API usage fees |
| Privacy | Fully local | Data sent to API provider |
