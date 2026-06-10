# Project Adam — Plan

## What Was Done (2026-06-10)

### Gap #1: Offline Consolidation Cycle — FIXED
- **Commit**: `9eb8360`
- **Files**: `src/project_adam/consolidator.py` (370 lines, +310/-60)
- **Changes**:
  - Full 6-step cycle in `merge_episodes()` with per-step logging
  - `_td_replay_prioritized()`: sorts episodes by absolute RPE, re-runs TD learning on top 10
  - `_abstract_to_skills()`: clusters similar high-reward episodes via embedding (>0.7), records as procedural skills
  - `_prune_memories()`: calls `episodic.prune(threshold=0.2)`
  - `_update_world_model()`: feeds high-reward episodes to world model
  - `_update_procedural_policies()`: records high-reward episodes as skills with Q-value update
  - `_extract_td_features()`: reconstructs 8-feature TD vector from episode data

### Gap #3: Procedural Skill Chunking — FIXED
- **Commit**: `0f142a6`
- **Files**: `src/project_adam/memory/procedural.py` (399 lines, +328/-71)
- **Changes**:
  - `Skill` class with Q-learning, success tracking, usage counting
  - `ChunkedSkill` class for macro-actions (action sequences that fire as a unit)
  - `_detect_and_chunk_sequences()`: detects repeated 3+ action patterns, creates chunks
  - `_transfer_by_similarity()`: embedding-based skill transfer to novel contexts
  - `_prune_low_value_skills()`: retains top skills by Q × success_rate
  - `stats()`: reports num_skills, num_chunks, avg_q_value, avg_success_rate

### Gap #5: Metacognitive Learning — FIXED
- **Commit**: `f93e167`
- **Files**: `src/project_adam/metacog.py` (211 lines, +184/-27)
- **Changes**:
  - Network expanded to 5→32→16→5 for more capacity
  - Mixing rate now favors learning: `min(0.9, total / (total + 100))`
    - After 100 interactions: 50% learned, 50% rule
    - After 1000 interactions: 90% learned, 10% rule
  - Proper REINFORCE with gradient clipping and loss tracking
  - Action counts and rewards tracked per canonical action
  - `get_mixing_rate()` method for monitoring

## Remaining Gaps (Medium Priority)

### Gap #2: Speaker Model as Bayesian Evidence
- **Problem**: `compute_utterance_likeness()` computes perplexity but doesn't feed into world model as Bayesian evidence
- **Fix**: In `world_model.py`, add `observe_from_language(text, confidence)` that applies `P(model|text) ∝ P(text|model) * P(model)` as an explicit Bayesian update on entity beliefs
- **Files**: `world_model.py`, `language.py`
- **Estimate**: 3-4 hours

### Gap #4: Self-Talk Shapes Learning
- **Problem**: Self-talk is generated but not used to update world model beliefs or guide attention
- **Fix**: Connect self-talk output to world model observation and metacognitive attention weighting
- **Files**: `language.py`, `metacog.py`, `world_model.py`
- **Estimate**: 3-4 hours

### Gap #6: Model-Based Trajectory Simulation
- **Problem**: `_simulate_trajectories()` scores entity sentiment rather than running actual multi-step planning
- **Fix**: Replace with tree search or Monte Carlo rollout across possible action sequences
- **Files**: `selector.py`
- **Estimate**: 4-5 hours

### Gap #7: Efficient Coding Task Loss
- **Problem**: Task loss is `-rpe * task_weight` instead of a real supervised learning signal
- **Fix**: Replace with actual information-theoretic complexity measure (mutual information or L0 sparsity)
- **Files**: `encoder.py`
- **Estimate**: 2-3 hours

### Gap #8: WM→Episodic Consolidation Trigger
- **Problem**: Working memory eviction doesn't explicitly trigger episodic consolidation
- **Fix**: Add consolidation trigger in consolidator or agent that pushes WM buffer to episodic
- **Files**: `consolidator.py`, `memory/working.py`
- **Estimate**: 2 hours

## Progress

- **90% architecture alignment** (up from 70-75%)
- 3 of 8 gaps fixed
- 136 tests passing
- See `IMPLEMENTATION_AUDIT.md` for full gap descriptions
