# DiffMemory — Differentiable Memory for COGNET

## Goal

Add a small differentiable memory module that learns compressed patterns from episodic experiences during consolidation, and retrieves relevant patterns during inference — without creating a parallel training path or adding heavy dependencies.

## Architecture compliance rule

> DiffMemory learns during `merge_episodes()` step 3 (Abstract), alongside existing embedding clustering. It retrieves during inference via a simple forward pass — no gradient computation. All data flows through the existing consolidation cycle. No parallel training path. No supervised bypass.

## Design

### The memory module

A 2-layer MLP that acts as a differentiable fast-weight memory:

```
MemoryMLP(384 → 1536 → 384):
  Linear(384, 1536) → GELU → Linear(1536, 384)
  LayerNorm(384) + residual
```

This accepts 384-dim embeddings (matching SentenceTransformer `all-MiniLM-L6-v2`) and projects them through a learned compression space. The MLP weights ARE the memory — storing patterns means updating weights via gradient descent.

### Store path (during consolidation)

```
merge_episodes() → _abstract_to_skills() → new step: _update_diffmemory()

_update_diffmemory(episodes):
  1. Filter high-reward episodes (reward > 0.4)
  2. Encode each via episodic_memory.embedder → 384-dim embeddings
  3. For each embedding:
     a. Compute reconstruction error: ‖MLP(emb) - emb‖²
     b. If error > threshold (0.15): this is a novel pattern → store it
     c. Store = single gradient step on MLP weights to minimize reconstruction error
  4. Consolidate: if MLP has > 200 stored patterns, prune least-used
```

### Retrieve path (during inference)

```
In chat(), after user input is encoded → retrieve matching patterns:

emb = episodic_memory.encode(user_input)
patterns = diffmemory.retrieve(emb, k=3)  # forward pass, no gradients
# patterns are text snippets reconstructed from the MLP
```

Retrieved patterns feed into `build_system_prompt()` as "What you know:" context — same path as semantic memory and spatial relations.

### No working memory injection

Retrieved patterns go into the system prompt, not working memory. This matches how all other long-term knowledge (semantic schemas, spatial relations, web results) is handled.

## Files changed

| File | Lines | Change |
|---|---|---|
| `src/project_adam/memory/diffmemory.py` | ~120 | **New** — `DiffMemory` class with MLP, store(), retrieve(), consolidate() |
| `src/project_adam/consolidator.py` | ~15 | Add `_update_diffmemory()` call in `merge_episodes()` |
| `src/project_adam/agent.py` | ~10 | Import + init + retrieval in `chat()` → `build_prompt()` |
| `src/project_adam/config.py` | ~8 | Config block for dim, threshold, max patterns |
| **Total** | **~153** | |

## Architecture compliance

| Principle | How satisfied |
|---|---|
| **RPE drives all learning** | Patterns stored during `merge_episodes()` — same cycle as RPE prioritization. No gradient updates during inference. |
| **No supervised bypass** | Store path goes through consolidation step 3. Retrieve is read-only forward pass. |
| **Multi-memory systems** | DiffMemory complements EpisodicMemory (exact storage) with compressed generalization. |
| **Language Interface** | Retrieved patterns feed into `build_prompt()`, same as semantic/spatial/web knowledge. |

## Implementation order

1. `src/project_adam/memory/diffmemory.py` — `DiffMemory` class
2. `src/project_adam/config.py` — config block
3. `src/project_adam/consolidator.py` — integration in step 3
4. `src/project_adam/agent.py` — init + retrieval
