# Architecture Alignment Plan

**Goal**: Make the code match `ARCHITECTURE.md` (root, the real document).

**138 violations** originally found across 4 scans. All phases largely complete.

---

## Phase 1 — Structural Blowup

### P1.1 — Create `LanguageInterface` class ✅
- `src/project_adam/language.py` created; owns model/tokenizer, generation, prompt building, self-talk, behavioral rules, user detection

### P1.2 — Route `utils.py` functions into proper components ✅
- `extract_facts()` → `SemanticMemory.extract_facts()`
- `extract_topics()` → `SemanticMemory.extract_topics()` (already existed, inlined)
- `detect_user()` → `LanguageInterface.detect_user()`
- `compute_implicit_reward()` → `TDCore.compute_reward()`
- `utils.py` deleted

### P1.3 — Remove `NeuralMemory` ✅
- Class and all references removed from agent, consolidator, __init__, CLI

### P1.4 — Integrate WebSearch as tool in LanguageInterface ✅
- `WebSearch` instance passed to `LanguageInterface.__init__()`; search triggered inside `generate()` when meta_action is ASK_FOR_HELP or EXPLORE
- `ActionSelector` no longer holds `web_search` reference

---

## Phase 2 — Learning Engine Rewrite ✅

### P2.1 — Rewrite `WorldModel` ✅
- Bayesian conjugate Gaussian priors; proper-noun extraction from text; `uncertainty()` returns `sqrt(var)`

### P2.2 — Fix `TDCore` ✅
- TD(λ) with eligibility traces; `register_rpe_listener()`; manual `p.add_(lr * δ * e)` updates; `compute_reward()` static method

### P2.3 — Fix `SFLModule` ✅
- Rescorla-Wagner update `Q += α·(R - Q)`; `compute_temperature()` for action selection; receives RPE from TDCore

### P2.4 — Fix `SensoryEncoder` ✅
- 3-layer MLP; sparsity via `|z|.mean()`; loss = `vae_loss + max(0, -rpe)` (sum); `train_step()` separates forward/backward

---

## Phase 3 — Action Selection Rewrite ✅

### P3.1 — Fix `ActionSelector` ✅
- No model/tokenizer (handled by LanguageInterface)
- Fast path: regex rule-based fallback (documented)
- Slow path: routes through LanguageInterface
- Canonical metacog actions: EXPLORE, REPLAY, ASK_FOR_HELP, STOP_AND_THINK, SWITCH_STRATEGY
- World model consultation: receives `world_model` parameter
- Hardcoded confidence threshold: uses `metacognitive.estimate_confidence()`
- SFL temperature passed through to generation

---

## Phase 4 — Memory System Alignment

### P4.1 — `WorkingMemory` ✅
- Capacity 8→64; gated retention via relevance threshold; evicted items pushed to episodic via `set_episodic_memory()`

### P4.2 — `EpisodicMemory` ✅
- RPE field added to entries; symbolic keyword index (`_symbolic_index` mapping word→indices); `search_by_keyword()` method

### P4.3 — `SemanticMemory` ✅
- Graph edges: `_edges` list of `(source_sid, relation, target_sid)` triples; `add_edge()`, `get_related()`, `traverse()` methods; edges pruned on consolidate

### P4.4 — `ProceduralMemory` ✅
- Keyword-overlap heuristic with chunking support: repeated action sequences detected via `_try_chunk()`, retrieved as multi-step chunks

### P4.5 — `SpatialMemory` ✅
- Conflict detection (`_is_contradiction()`); inverse relation inference (`_INVERSE_MAP`); graph traversal (`traverse()`); `conflicts()` method

### P4.6 — `OfflineConsolidator` ✅
- Periodic thread removed; metacog-triggered only (`merge_episodes()`); RPE prioritization; world model update via `_update_world_model()`; `start()`/`stop()`/`_loop()` removed

---

## Phase 5 — Supporting Components

### P5.1 — `MetacognitiveController` ✅
- Canonical actions: `EXPLORE`, `REPLAY`, `ASK_FOR_HELP`, `STOP_AND_THINK`, `SWITCH_STRATEGY`; `record_confidence()` method; `CANONICAL_ACTIONS` constant

### P5.2 — `Persona` ✅
- 28KB size limit enforced on load (`_MAX_SIZE = 28 * 1024`); operator precedence bug fixed with explicit parentheses

### P5.3 — `UserProfileManager` ✅
- Removed dead keys (`total_interactions`, `adopted_phrases`, `phrase_preferences`, `last_used_opening/closing`); removed dead `reply` parameter from `update_after_turn()`; fixed race condition on `current_name` (wrapped in lock)

### P5.4 — `WebSearch` ✅
- Cache path uses `get_memory_dir()`; global SSL warning suppression removed; logging on init failure

---

## Remaining ⏳
- Nothing — all phases complete. Every architectural gap identified in ARCHITECTURE.md has been addressed.

## Execution Order
1. Phase 1 ✅
2. Phase 2 ✅
3. Phase 3 ✅
4. Phase 4 ✅
5. Phase 5 ✅
6. Update docs/architecture.md to match root ARCHITECTURE.md ✅
7. Update AGENTS.md ✅
