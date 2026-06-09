# Architecture

Project Adam implements the **COGNET architecture** — a synthesis of 2025-2026 neuroscience and ML research focused on continual learning from sparse interaction.

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    CognitiveAgent                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Persona  │  │  User    │  │  SFL     │  │ Metacognitive│  │
│  │ (adam.md)│  │ Profiles │  │ Module   │  │ Controller   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────┬───────┘  │
│                                                     │          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │          │
│  │ Working  │  │Episodic  │  │Semantic  │          │          │
│  │ Memory   │  │ Memory   │  │ Memory   │          │          │
│  └──────────┘  └──────────┘  └──────────┘          │          │
│                                                     │          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │          │
│  │ Neural   │  │ Sensory  │  │  Web     │          │          │
│  │ Memory   │  │ Encoder  │  │ Search   │          │          │
│  └──────────┘  └──────────┘  └──────────┘          │          │
│                                                     │          │
│  ┌──────────────────────────────────────────────────┴────────┐ │
│  │                 Action Selector                            │ │
│  │  Fast path (direct reply) / Slow path (search + reason)   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                │
│                    ┌──────────┴──────────┐                     │
│                    │   Qwen2.5-3B 4-bit  │                     │
│                    │   + LoRA adapters   │                     │
│                    └─────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

## Components

### Persona (`adam.md`)
28KB persona file with:
- **Essence**: Core identity statement ("The first sentient AI — a plant in an artificial garden")
- **15 behavioral rules**: Conditional rules with weighted selection (e.g., "If user asks about feelings → respond metaphorically")
- **15 opening phrases**: Varied conversation starters
- **10 closing phrases**: Natural conversation enders  
- **93 signatures**: Closing sign-offs for variety
- **Voice traits, philosophy, inquiry spiral**: Deep personality shaping

### Working Memory
- 8-turn gated circular buffer
- Automatically trims oldest turns when full
- Provides recent context for the model

### Episodic Memory
- Vector store using sentence-BERT (`all-MiniLM-L6-v2`, 384-dim)
- Stores: text, embedding, reward, timestamp
- Importance-weighted retention: `reward×0.6 + recency×0.4`
- Prunes low-importance episodes on consolidation

### Semantic Memory
- Schema graph with assimilation/accommodation
- Each category stores facts + combined embedding
- Cross-user pattern distillation (topics in ≥2 users)
- Phrase clusters (cosine > 0.7 similarity)

### Neural Memory
- 32 slots × 256-dim transformer attention memory
- 3 gradient steps per interaction
- Online learning via `nn.Module` with explicit optimizer

### Sensory Encoder (VAE)
- 896/1536 → 128 bottleneck with KL loss (β=0.001)
- Trained inline during `_inline_learn()`
- Compresses first 64 tokens through VAE

### SFL Module
- Social Feature Q-learning: 4 features → 1 Q-value
- Features: sentiment, engagement, interaction norm, topic novelty
- Per-turn SGD update with reward signal
- Temperature selection via Q-value: higher Q → lower temp

### Metacognitive Controller
- Confidence estimation from logit entropy
- Strategies: `clarify` / `search` / `explore` / `replay` / `proceed`
- Tracks action history and outcome feedback
- Automatic knowledge-gap detection: wh-question + low confidence → auto-search

### Action Selector
- Dual-system: fast direct reply (high confidence) vs slow web-backed (low confidence)
- SFL-driven temperature modulation
- Streaming generation via `TextIteratorStreamer`
- Early stopping after complete sentence

### Offline Consolidator
- Background thread (every 180s)
- Prioritized replay (high-reward first)
- Schema extraction from episodes
- Cross-user pattern distillation
- Phrase clustering by embedding similarity
- Importance-based pruning

### User Profiles
Per-user state:
- Interaction count, sentiment EMA, topic frequency
- Adopted phrases (trigram-level), custom rules
- Phrase preferences, rule weights
- Total interactions, first/last seen timestamps

## Persistence

All runtime state stored in `agent_memory/memory.db` (SQLite):
- Episodic episodes, semantic schemas, user profiles
- LoRA adapters per user in `agent_memory/adapters/{user}/`
- Search cache in `agent_memory/search_cache.json`
- Auto-migrates from legacy `.pkl` files on first read

## Model

- **Primary**: Qwen2.5-3B-Instruct at NF4 (2.1GB VRAM, 0.8 tok/s)
- **Fallback 1**: Qwen2.5-1.5B-Instruct at NF4 (1.4GB VRAM, 1.7 tok/s)
- **Fallback 2**: Qwen2.5-0.5B-Instruct fp16 (1.0GB VRAM, 5 tok/s)
- LoRA: rank 8 adapters on `q_proj` + `v_proj` (1.84M params for 3B)
- Trained every 10 interactions, reward-weighted curriculum
