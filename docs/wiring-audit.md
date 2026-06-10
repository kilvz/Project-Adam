# Wiring Audit — Unwired Code

**Date**: 2026-06-10  
**Method**: Read all 27 source files, traced every method call, field read/write.

## Must-Wire (architecture explicitly requires these data flows)

### 1. SpatialMemory — query, traverse, consolidate
- `query()`: entity lookup never used
- `traverse()`: graph traversal never used
- `consolidate()`: never cleaned up → grows unbounded
- **Fix**: Wire into `selector._slow_path()` and `OfflineConsolidator.merge_episodes()`

### 2. WorldModel — consolidate, simulate
- `consolidate()`: entities grow unbounded, never pruned
- `simulate()`: what-if prediction exists but dead code
- **Fix**: Wire consolidate into consolidation cycle; wire simulate into trajectory scoring

### 3. SemanticMemory — add_edge, traverse
- `add_edge()`: graph edges never created
- `traverse()`: graph navigation dead
- **Fix**: Call add_edge inside _assimilate(); use traverse during retrieve()

### 4. Persona — build_user_prompt
- User context builder dead code
- **Fix**: Call from LanguageInterface.build_prompt()

### 5. ProceduralMemory — record_failure
- Negative reward never recorded
- **Fix**: Call from agent.chat() when reward < 0

### 6. WorkingMemory — set_goal, push_hypothesis, pop_hypothesis
- Goal tracking and hypothesis stack never used
- **Fix**: Wire into metacognitive strategy decisions

## Cleanup (dead code removal)

### 7. selector.py — unused fields
- `self.episodic`, `self.semantic` stored but never read

### 8. rl_core.py — dead methods
- `register_rpe_listener()`: no callers
- `predict()`: no callers
- `reset()`: no callers

### 9. metacog.py — dead methods and fields
- `should_search()`: no callers
- `self.strategy_history`: written, never read
- `self.slow_path`: set to 0, never used

### 10. encoder.py — stubs
- `VisionEncoder`, `AudioEncoder`: never instantiated
- `compute_complexity()`: no callers
- `vae_loss()`: no callers

### 11. memory/episodic.py — search_by_keyword, recent
- `search_by_keyword()`: symbolic index lookup never used
- `recent()`: no callers

### 12. memory/semantic.py — cross_user_distill, phrase_cluster
- `cross_user_distill()`: superseded by consolidator's own version
- `phrase_cluster()`: superseded by consolidator's own version

### 13. memory/working.py — clear, get_gated_context
- `clear()`: no callers
- `get_gated_context()`: no callers

### 14. profiles.py — remove
- `remove()`: no callers

## Execution Plan

### Phase 1 — Wire architecture-required connections
1. Wire SpatialMemory query/traverse into `_slow_path()`
2. Wire SpatialMemory consolidate into `merge_episodes()`
3. Wire WorldModel consolidate into `merge_episodes()`
4. Wire WorldModel simulate into `_simulate_trajectories()`
5. Wire SemanticMemory add_edge inside `_assimilate()`
6. Wire Persona build_user_prompt into `build_prompt()`
7. Wire ProceduralMemory record_failure into `chat()`
8. Wire WorkingMemory goal/hypothesis into metacog

### Phase 2 — Remove dead code
9. Remove unused selector fields
10. Remove unused rl_core methods
11. Remove unused metacog methods/fields
12. Remove unused encoder stubs (keep as doc)
13. Remove unused memory methods
