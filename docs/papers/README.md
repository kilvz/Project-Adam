# Papers — Reference Index

Papers referenced by Project Adam's COGNET architecture and DiffMemory implementation.

## Core Memory Papers

| Paper | File | Topic |
|-------|------|-------|
| Titans (2501.00663) | [titans.md](titans.md) | Neural long-term memory module, surprise-gated storage, momentum, adaptive forgetting |
| MIRAS (2504.13173) | [miras.md](miras.md) | Unification of sequence models, attentional bias, retention gates, YAAD/Huber loss |

## How They Map to Our Architecture

```
Titans Neural Memory ──────────→ DiffMemory (memory/diffmemory.py)
  ├─ MLP as memory              ✅ _MemoryMLP (2-layer, depth configurable)
  ├─ Surprise metric             ✅ Reconstruction error (v3: gradient magnitude)
  ├─ Momentum                    ✅ _momentum buffer
  └─ Adaptive forgetting         ✅ AdamW + re-encode prune (v3: learned decay head)

MIRAS Framework ───────────────→ DiffMemory design choices
  ├─ Attentional bias (MSE)      ✅ MSE loss (v3: +Huber/YAAD option)
  ├─ Retention gate              ✅ Weight decay (v3: +learned head)
  └─ Memory algorithm            ✅ AdamW optimizer
```

## Architecture Papers (COGNET)

Referenced in `architecture.md`:

| Principle | Paper |
|-----------|-------|
| Efficient coding | Nature Communications 2025 |
| Unified RL mechanism | Nature Human Behaviour 2025 |
| Multi-memory systems | Cognitive Science, HAMI 2025 |
| Active retrieval | Karpicke & Bjork |
| Prior knowledge integration | How People Learn (NRC) |
| Dual-system architecture | ICML 2025 |
| Offline consolidation | Bio-realistic Hippocampus 2025 |
| Metacognition | APA Top 20, Karpicke |
| Language as accelerator | OpenReview 2026 |
| Individual latent states | Frontiers Neuroscience 2026 |
