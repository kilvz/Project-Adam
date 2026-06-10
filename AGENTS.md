# Project Adam — Agent Instructions

## Golden Rules

### 1. Plan first, code second
Before writing any code, create or update `docs/plan.md` with a clear plan of what will be changed and why. Present it to the user before executing.

### 2. Keep the plan current
Update `docs/plan.md` as work progresses — mark done items, add new steps, note blockers. The plan is the single source of truth for what's happening.

### 3. Update AGENTS.md
When you learn something about the project (conventions, quirks, decisions), record it here so future sessions benefit.

### 4. Document everything
All non-trivial changes must be reflected in the `docs/` folder. Update existing docs or create new ones as needed.

### 5. Zero-tolerance architecture compliance
This project uses the **COGNET architecture** defined in `architecture.md` (root). Every component, interface, and data flow MUST match the architecture spec exactly.

**Rules:**
- `architecture.md` is the absolute ground truth. No simplifications, no shortcuts, no tradeoffs.
- Every component must implement ALL responsibilities listed in its architecture section.
- Every signal and data flow in the architecture diagrams must be present in the code.
- The mathematical formulas in the architecture (loss functions, update rules, Bayesian equations) must be implemented literally — not approximated.
- If a component is described as "a separate small network", it must be an actual neural network, not a rule-based heuristic.
- If the architecture says "learned via RL", the component must learn through an RL signal (RPE, TD error, policy gradient, etc.), not through heuristics.
- "Language as evidence" means probabilistic evidence integration (e.g., speaker model, confidence-weighted observations), not just entity extraction.
- The consolidation cycle must faithfully implement all 6 steps (Replay → Prioritize → Abstract → Prune → Update world model → Update procedural policies).
- **Never** leave a gap as a "noted tradeoff" — fix it. If VRAM or compute is a constraint, find a smaller version of the correct approach rather than a different approach.
- Before marking any task done, verify the implementation against every sentence of the relevant architecture section. If the architecture says it, the code must do it.

### 6. Stick to the package structure
All source code lives under `src/project_adam/`. Tests in `tests/`. Never dump loose scripts in the root.

## Architecture Alignment (COGNET)

The implementation makes deliberate tradeoffs to fit 4GB VRAM (GTX 1050):

| Theoretical Component | Implementation | Tradeoff |
|---|---|---|
| **Sensory Encoder (VAE)** | `SensoryEncoder` with β-VAE, RPE-weighted loss, 3-layer MLP | Single modality (text); no learned prior |
| **Working Memory** | `WorkingMemory` with relevance-gated eviction, 64 slots, episodic push | Embedder is expensive per-add; no learned gating |
| **Episodic Memory** | `EpisodicMemory` with SentenceTransformer + SQLite, symbolic keyword index | FAISS would be faster but SQLite works at <1K episodes |
| **Semantic Memory** | `SemanticMemory` with assimilation/accommodation, graph edges between schemas | True graph DB (Neo4j) would scale better; dict is fine for personal use |
| **Procedural Memory** | `ProceduralMemory` with keyword-overlap matching, success-rate tracking | Not learned skills; heuristic pattern matching |
| **Spatial Memory** | `SpatialMemory` with directed triple store (17 relations, 200 cap), conflict detection, inverse inference, graph traversal | No full spatial reasoning |
| **RL Core (TD(λ))** | `TDCore` with linear ValueNetwork, eligibility traces via manual RPE broadcast | Linear value function, not function approximation |
| **World Model** | `WorldModel` with conjugate Gaussian priors, proper-noun extraction, observation | Not full Bayesian inference; no causal graph |
| **SFL Module** | `SFLModule` Rescorla-Wagner update (no GD), `compute_temperature()` for action selection | Matches architecture; low-dim features (4) |
| **Metacognitive Controller** | `MetacognitiveController` with canonical actions: EXPLORE, REPLAY, ASK_FOR_HELP, STOP_AND_THINK, SWITCH_STRATEGY | Outputs string actions, not learned control policy |
| **Action Selection** | `ActionSelector` dual-system: pattern fast path + model slow path, world model consultation | Fast path is regex, not model-free RL; no true planning |
| **Offline Consolidator** | `OfflineConsolidator` with RPE prioritization, world model update, clustering | Metacog-triggered only (not periodic); no full sleep cycle |

## Key Fixes Applied (2026-06-10)

### TD Update called twice per chat turn
**File**: `agent.py`  
**Bug**: `self.td_core.update(reward, td_features)` was called twice — once at line 187 and again at line 221. The second call computed TD error with `V(s) == V(s')` (same features), producing `reward + (γ-1)·V(s)` instead of `reward + γ·V(s') - V(s)`.  
**Fix**: Removed the second call. Single update per turn is correct.

### Redundant internal state override
**File**: `agent.py`  
**Bug**: `self.td_core.last_features = td_features` manually set an internal attribute that `update()` already sets. Redundant and violated encapsulation.  
**Fix**: Removed. `TDCore.update()` owns its own state.

### Direct metacognitive state mutation
**File**: `agent.py`  
**Bug**: Agent bypassed metacog's interface to directly append to its deque.  
**Fix**: Added `record_confidence()` method to `MetacognitiveController`; agent calls that instead.

### Dead attribute assignment
**File**: `agent.py`  
**Bug**: Set a nonexistent attribute that was never read.  
**Fix**: Removed.

### Self-talk generation in wrong component
**File**: `metacog.py`  
**Bug**: MetacognitiveController generated natural language text. Per architecture, it should output **metacognitive actions** (strings like `"low_confidence"`, `"unfamiliar"`). Text generation belongs to the Language Interface (agent).  
**Fix**: Renamed to `should_self_talk()` returning a reason string; agent generates the actual text.

### SFL double-update per turn (2026-06-10 continuation)
**File**: `agent.py` (removed `_on_rpe` listener registration)  
**Bug**: `TDCore.update()` broadcast RPE via `_on_rpe` listener, then lines 180-182 explicitly called `sfl_module.update()` again. Each turn produced two SFL updates with different features (empty string vs real input).  
**Fix**: Removed `_on_rpe` listener. Single explicit `sfl_module.update()` per turn with proper features.

### RL Core gradient shape mismatch (2026-06-10)
**File**: `rl_core.py:56`  
**Bug**: `v_s` had shape `[1,1]` (batched). `torch.autograd.grad(v_s, params)` produced batched gradients that didn't match eligibility trace shapes for bias parameters, causing RuntimeError.  
**Fix**: Added `.squeeze(-1)` so `v_s` is `[1]`; autograd yields per-parameter gradients.

### WorldModel dead method (2026-06-10)
**File**: `world_model.py:37-38`  
**Bug**: `observe_from_text()` was a no-op (`pass`), so world model was never populated despite being called every turn.  
**Fix**: Implemented proper-noun extraction with regex and `observe(entity, "sentiment", value)`.

### OfflineConsolidator periodic thread removed (2026-06-10)
**File**: `consolidator.py`  
**Bug**: A background thread consolidated every 180s regardless of metacog state; the architecture says consolidation is metacog-triggered.  
**Fix**: Removed `_loop()` thread, `start()`, `stop()`. Consolidation only runs on `merge_episodes()` called by REPLAY meta-action. Uses RPE for prioritization, updates world model.

### NeuralMemory references in CLI (2026-06-10)
**File**: `ui/cli.py:24,99`  
**Bug**: After NeuralMemory was deleted in Phase 1, CLI still referenced `agent.neural_memory.state_dict()`, causing AttributeError on exit.  
**Fix**: Removed the two stale references.

### utils.py deleted (2026-06-10)
**File**: `utils.py`  
**Bug**: `extract_facts`, `compute_implicit_reward`, `detect_user`, `extract_topics` lived in a catch-all module with no architectural role.  
**Fix**: `extract_facts` → `SemanticMemory.extract_facts()`. `compute_implicit_reward` → `TDCore.compute_reward()`. `detect_user` → `LanguageInterface.detect_user()`. `extract_topics` → `SemanticMemory.extract_topics()` (already existed, inlined). `utils.py` deleted.

## CUDA Setup

- **Driver**: nvidia-driver-580 (580.159.03) — NOT 550
- **CUDA**: 13.0
- **GPU**: GTX 1050 (4GB VRAM, sm_61 Pascal)
- **Torch**: 2.6.0
- **VRAM budget**: 2.11GB idle (3B 4-bit), 2.15GB generating (within 4GB)

If nvidia-smi is missing: `sudo apt install nvidia-driver-580 && sudo reboot`

## Test Commands

```bash
PYTHONPATH=src python3 -m pytest tests/ -v            # all tests
PYTHONPATH=src python3 -m pytest tests/test_X.py -v   # single file
```
