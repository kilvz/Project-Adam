# Memory System

Adam uses three complementary memory systems inspired by human cognition.

## 1. Episodic Memory

Stores conversation experiences as vectors.

```
EpisodicMemory (SQLite-backed)
├── encode(text) → 384-dim embedding (all-MiniLM-L6-v2)
├── store(episode, reward) → SQLite insert
├── recall(query, top_k=5) → similarity search
├── recent(n=10) → last N episodes
└── prune(threshold=0.3) → low-importance cleanup
```

- **Storage**: SQLite table `episodes` with embedding + metadata + reward
- **Retrieval**: Cosine similarity via numpy dot product (full scan, ~5ms per 1000 episodes)
- **Importance weighting**: `reward * 0.6 + recency * 0.4`
- **Consolidation**: Prunes episodes below 0.3 importance; keeps top 100 recent

## 2. Semantic Memory

Schema-based knowledge graph.

```
SemanticMemory (SQLite-backed)
├── Schema: category → {facts[], combined_embedding, observed_count}
├── store_schema(category, fact) → assimilate or accommodate
├── consolidate() → prune + merge similar schemas
├── extract_topics(text) → keyword extraction
├── cross_user_distill(profiles) → patterns in ≥2 users
└── phrase_cluster(threshold=0.7) → cluster by cosine similarity
```

- **Assimilation**: New fact fits existing schema → added
- **Accommodation**: New fact requires schema restructuring
- **Cross-user distillation**: Topics mentioned by ≥2 users are flagged as "distilled"
- **Phrase clustering**: Trigram-level, grouped by embedding similarity > 0.7

## 3. Neural Memory

Gradient-updated attention memory with explicit optimizer.

```
NeuralMemory (nn.Module)
├── slots: 32 × 256-dim
├── update(text_embedding) → 3 gradient steps
├── read(query) → attention-weighted slot sum
└── consolidate() → reset unused slots
```

- **Implementation**: `nn.Module` with `nn.Parameter` memory matrix
- **Update rule**: Cosine similarity attention → weighted combination → gradient descent
- **Online**: Trained during `_inline_learn()` alongside VAE
- **Architecture**: 384 → 256 projection, 256 → 256 transformer block, 256 → 128 output

## SQLite Backend (Phase E.23)

All three memory systems (plus user profiles) persist via `SQLiteStore`:

```
agent_memory/memory.db
├── episodes (episodic)
├── schemas (semantic)
├── neural_memory (serialized state dict)
├── user_profiles (per-user state)
├── episodes_meta (metadata index)
└── schema_facts (fact storage)
```

- WAL mode for concurrent reads
- Atomic transactions
- Lazy auto-migration from legacy `.pkl` files
- All data survives restarts

## Offline Consolidator

Background thread (every 180s) that:
1. Samples episodes by importance-weighted priority
2. Extracts schemas from high-reward experiences
3. Distills cross-user patterns
4. Clusters phrases by embedding similarity
5. Prunes low-importance episodes and schemas
