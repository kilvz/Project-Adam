# DiffMemory v2 — Titans-Aligned Upgrades

## Goal

Upgrade DiffMemory to incorporate three key mechanisms from the Titans architecture: momentum-based surprise tracking, weight decay forgetting, and configurable memory depth — without changing the architecture-compliant training path (consolidation step 3b, no gradients during inference).

## Changes

### 1. Momentum-based surprise tracking

**Current**: `store()` computes reconstruction error per-token. If error < threshold, skip. Each token evaluated independently — loses sequential pattern.

**New**: Maintain a momentum buffer that carries surprise across tokens in a batch:

```
momentum = β · momentum_prev + (1 - β) · surprise
effective_surprise = surprise + α · momentum
```

Where `β = momentum_beta` (default 0.9) and `α = momentum_scale` (default 0.5). If token N has high surprise, momentum stays elevated for subsequent tokens — so token N+1 (individually less surprising but contextually related) also gets stored.

### 2. Weight decay forgetting

**Current**: LRU eviction — sorts `self.patterns` by `usage_count`, drops lowest when over `max_patterns`. Heuristic, not learned.

**New**: Two changes:
- Replace `Adam` with `AdamW` (decoupled weight decay) — the optimizer naturally decays unused weight associations
- Replace LRU eviction with **re-encode pruning**: periodically re-encode all stored pattern texts through the MLP, drop those with high reconstruction error (meaning the MLP has "forgotten" them — they're no longer represented in the weights)

The pattern list stays as a **text cache** for retrieval display, but memory capacity is managed by the MLP weights, not the pattern list.

### 3. Configurable memory depth

**Current**: Hardcoded 2-layer MLP.

**New**: Accept `depth` parameter (default 2, backward compatible):

| Depth | Architecture | VRAM |
|-------|-------------|------|
| 2 | 384→1536→384 | ~3 MB |
| 4 | 384→1536→1536→1536→384 | ~6 MB |
| 6 | 384→1536→1536→1536→1536→1536→384 | ~9 MB |

Per Titans paper: deeper memory → better perplexity, better scaling to long sequences.

## Files changed

| File | Lines | Change |
|---|---|---|
| `memory/diffmemory.py` | ~60 | Momentum buffer, AdamW, depth, re-encode pruning |
| `config.py` | ~5 | Add `momentum_beta`, `momentum_scale`, `weight_decay` to `DIFFMEMORY_CONFIG` |

## VRAM impact

| Addition | VRAM |
|---|---|
| Momentum buffer (1 float, no graph) | ~0 MB |
| AdamW → Adam (same optimizer state size) | +0 MB |
| Deeper MLP (2→4 layers) | ~+3 MB |
| **Total** | **~+3 MB** |

## Architecture compliance

| Principle | How satisfied |
|---|---|
| **No gradients during inference** | Momentum and weight decay only affect `store()`, never `retrieve()` |
| **RPE drives all learning** | DiffMemory still learns during consolidation step 3b |
| **No new dependencies** | AdamW is built into PyTorch (`torch.optim.AdamW`) |
