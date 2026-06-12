# Memory Systems

Adam has seven memory systems as specified by the COGNET architecture.

## Working Memory

**File**: `src/project_adam/memory/working.py`

Bounded-capacity scratchpad for current context, goals, and active reasoning.

- **Capacity**: 64 slots (turns), 2048 token window
- **Gating**: Relevance computed via scaled dot-product attention (SentenceTransformer embeddings). Items below `relevance_threshold` bypass WM and go directly to episodic memory.
- **Decay**: Temporal decay (`tau=300s`) applied to relevance scores on eviction
- **Content**: Dialogue turns (`user`/`adam`), active `goal` string, `hypothesis_stack`
- **Eviction**: Lowest decayed-relevance item pushed to episodic memory

## Episodic Memory

**File**: `src/project_adam/memory/episodic.py`

Long-term storage of experiences as `(state, action, reward, context)` tuples.

- **Storage**: SQLite with pickle serialization. Entry format: `{text, state, action, reward, rpe, context, backend, ts, emb, count}`
- **Indexing**: Symbolic keyword index (`word → [episode_indices]`) for fast retrieval
- **Compression**: Episodes with cosine similarity > 0.92 are merged (reward averaged, count incremented)
- **Search**: Content-addressable via SentenceTransformer embedding cosine similarity. Also supports `search_by_keyword()` for symbolic lookup.
- **Backend tracking**: Each episode records which backend generated the action (`"local"` or `"api"`) for downstream distillation analysis.
- **Latent codes**: Encoder latent `z` stored as `latent_z` for future representation learning.

## Semantic Memory

**File**: `src/project_adam/memory/semantic.py`

Graph-structured knowledge base of concepts and relations.

- **Schemas**: Each concept is a schema with `{category, facts[], emb, slots{}, prediction_error, observed_count}`
- **Graph**: Directed edges via `_edges` list of `(source_sid, relation, target_sid)` triples
- **Assimilation**: New facts similar to existing schemas (>0.75 cosine) are assimilated — facts appended, slots updated, edges added
- **Accommodation**: Novel facts create new schemas. Schema splitting when internal distance > 0.8
- **Text extraction**: `extract_facts()` and `extract_topics()` for parsing user input
- **Consolidation**: Low-observation schemas pruned periodically

## Procedural Memory

**File**: `src/project_adam/memory/procedural.py`

Stores learned skills as `(state → action)` mappings with Q-values.

- **Skills**: Keyword sets mapped to action strings with `q_value`, `success_count`, `total_count`
- **RL**: Q-values updated via RPE from TDCore (`update_from_rpe()`)
- **Chunking**: Repeated action sequences detected and stored as chunks (min 2 occurrences)
- **Retrieval**: Scores skills by keyword overlap × Q-value. Chunks preferred over individual skills when overlap matches.
- **Failure tracking**: `record_failure()` called when reward < 0

## Spatial Memory

**File**: `src/project_adam/memory/spatial.py`

Dynamic knowledge graph of spatial relationships.

- **Triples**: `(entity_a, relation, entity_b)` with 17 supported relations
- **Inverse relations**: Automatic insertion of inverse (above↔below, inside↔contains, etc.)
- **Conflict detection**: Contradictory pairs detected on insertion (above↔below, near↔far, etc.)
- **Traversal**: BFS traversal up to 3 hops for spatial reasoning
- **Text extraction**: Regex pattern matching for `entity RELATION entity` triples in free text

## User Profile Memory

**File**: `src/project_adam/profiles.py`

Per-user persistent state stored in SQLite.

- **Data**: interaction_count, avg_sentiment, topics, custom_rules, rule_weights, sentiment_history
- **Detection**: Automatic user detection via name extraction from conversation
- **Adapters**: Per-user LoRA adapters saved at `agent_memory/adapters/{user}/`

## DiffMemory

**File**: `src/project_adam/memory/diffmemory.py`

Lightweight differentiable memory that learns compressed patterns from episodic experiences during consolidation.

- **Architecture**: 2-layer `MemoryMLP` (384→1536→384) with GELU activations, LayerNorm + residual
- **Storage**: Patterns stored via gradient descent on MLP weights — the MLP weights ARE the memory
- **Surprise gating**: Only stores patterns with reconstruction error above `surprise_threshold` (0.15) — prevents duplicate storage
- **Retrieval**: Forward pass through MLP, then cosine similarity against stored pattern embeddings
- **Training**: Updated during consolidation step 3b (`_update_diffmemory`), not during inference
- **Inference**: No gradient computation — pure forward pass + nearest-neighbor
- **Capacity**: Bounded by `max_patterns` (200) with LRU eviction based on usage count
- **VRAM**: ~3 MB total (MLP weights + gradient buffers during training)
- **Integration**: Patterns retrieved in `chat()` and injected into `build_system_prompt()` via `user_profile["memory_patterns"]`
