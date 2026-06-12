# COGNET — Brain-Inspired Cognitive AI Architecture

Design synthesized from 2025-2026 research on human learning and behavior.

## Core Principles

| # | Principle | Source | Implication |
|---|-----------|--------|-------------|
| 1 | Efficient coding | Nat. Comms 2025 | Minimize representation complexity while maximizing reward |
| 2 | Unified RL mechanism | Nat. Hum. Behav. 2025 | Same reward-learning drives both individual & social learning |
| 3 | Multi-memory systems | Cogn. Sci., HAMI 2025 | Sensory → Working → Long-term (episodic + semantic + procedural) |
| 4 | Active retrieval | Karpicke, Bjork | Retrieval practice > passive storage |
| 5 | Prior knowledge integration | How People Learn (NRC) | New info interpreted through existing schemata |
| 6 | Dual-system architecture | ICML 2025 | Model-free (fast/habit) + Model-based (slow/deliberate) |
| 7 | Offline consolidation | Bio-realistic Hippocampus 2025 | Experience replay, prioritization, abstraction during "sleep" |
| 8 | Metacognition | APA Top 20, Karpicke | Self-monitoring of uncertainty, confidence, and progress |
| 9 | Language as accelerator | OpenReview 2026 | Linguistic guidance shapes exploration and reduces search |
| 10 | Individual latent states | Frontiers Neurosci. 2026 | Differences in how context is carved into states |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    METACOGNITIVE CONTROLLER                  │
│  uncertainty estimation · confidence monitoring              │
│  strategy selection · self-evaluation · reflection           │
└────────────┬───────────────┬───────────────────┬────────────┘
             │               │                   │
     ┌───────┴───────┐ ┌────┴────┐ ┌─────────────┴──────────────┐
     │   SENSORY     │ │ WORKING │ │      LANGUAGE INTERFACE    │
     │   ENCODERS    │ │ MEMORY  │ │  (generation, self-talk,   │
     │ (text/vision/ │ │(64-slot,│ │   behavioral rules,        │
     │  audio β-VAE) │ │ gated)  │ │   utterance-likeness)      │
     └───────┬───────┘ └────┬────┘ └─────────────┬──────────────┘
             │              │                    │
             ▼              ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LONG-TERM MEMORY SYSTEMS                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐         │
│  │ Episodic │ │ Semantic │ │Procedural│ │  Spatial   │         │
│  │ (events, │ │(schemata,│ │ (skills, │ │ (layouts,  │         │
│  │ contexts)│ │  graphs) │ │ policies)│ │ relations) │         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘         │
│       │            │            │              │                │
│  ┌────┴────────────┴────────────┴──────────────┴──────┐         │
│  │               OFFLINE CONSOLIDATOR                 │         │
│  │  replay · prioritize · abstract · prune · update   │         │
│  │  world model · update procedural · DiffMemory      │         │
│  └────────────────────────┬──────────────────────────-┘         │
│                           │                                    │
│  ┌────────────────────────┴──────────────────────────────┐     │
│  │  DiffMemory (differentiable MLP, consolidation-trained) │     │
│  │  surprise-gated storage, fast-forward retrieval         │     │
│  └────────────────────────────────────────────────────────-┘     │
│                    │                                             │
│  ┌─────────────────┴─────────────────────────────────────┐      │
│  │  WebSearch (DDGS + Wikipedia, cached)                 │      │
│  │  Self-Play (background Q→A generation, metacog-gated) │      │
│  │  PersonaManager (list, load, switch, generate)        │      │
│  └──────────────────────────────────────────────────────-┘      │
└──────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    LEARNING ENGINE                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ RL Core  │ │ Efficient│ │  Social  │ │   Bayesian   │  │
│  │ (TD-     │ │ Coding   │ │ Feature  │ │  World Model │  │
│  │ learning)│ │ Objective│ │ Learning │ │   Inference  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    ACTION SELECTION (Dual-System)            │
│  ┌────────────────────┐  ┌──────────────────────────────┐   │
│  │ Model-Free (fast)  │  │ Model-Based (slow/deliberate)│   │
│  │ habitual policies  │  │ planning · search · reasoning│   │
│  └────────────────────┘  └──────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Sensory Encoders (Efficient Coding)

**Purpose:** Convert raw input into minimal sufficient representations.

```
Input → Encoder → Information Bottleneck → Sparse Latent Code
```

- Each modality (vision, text, audio) has a dedicated encoder
- Encoder is trained with a **rate-distortion objective**: minimize `I(X;Z)` (mutual info between input and representation) while maximizing task performance
- Implemented as VAE with a learned prior or a sparse autoencoder with L0 regularization
- Produces a **sparse, factorized latent code** `z` where only ~5-10% of dimensions are active at any time
- This mirrors the brain's efficient coding — the SFL model's insight that humans distill stimuli into compact representations

**Mathematical objective:**
```
L_enc = E[reward] - β · I(X;Z)
```
where β controls the compression strength (higher β → more compression).

### 2. Working Memory

**Purpose:** Bounded-capacity scratchpad for current context, goals, and active reasoning.

```
Structure: Sliding Window Attention + Explicit Bounded Buffer
- Window size: K tokens/steps (e.g., 2048)
- Buffer capacity: fixed at C slots (e.g., 64)
- Attention gates what enters working memory
- Items decay and are pushed to episodic memory when displaced
```

- Based on the cognitive science finding that working memory is limited (~7±2 chunks)
- Uses a transformer core with **gated retention** — not all inputs enter WM
- WM holds: current observation, active goal, hypothesis stack, recent action history
- When WM is full, oldest/least-relevant items are **consolidated** into episodic memory

### 3. Long-Term Memory Systems

#### 3a. Episodic Memory
- Stores sequences of (state, action, reward, context) tuples
- Indexed by **symbolic keys** (like HAMI's symbolic indexing) for fast retrieval
- Supports **temporal compression** — repeated patterns are abstracted rather than stored redundantly
- Retrieval via content-addressable search (nearest neighbor in a learned embedding space)

#### 3b. Semantic Memory (Schemata)
- Graph-structured knowledge base of concepts and relations
- Each concept is a **schema** — a structured bundle of features with slot values
- New information is integrated by:
  1. Finding the best-matching existing schema(s)
  2. Computing prediction error (mismatch between expectation and observation)
  3. If error is small → assimilate into existing schema (update slots)
  4. If error is large → create new schema or split existing one (accommodation)
- This mirrors Piaget's assimilation/accommodation and the "Fish Is Fish" effect from *How People Learn*

#### 3c. Procedural Memory
- Stores policies (state → action mappings) as reusable skills
- Learned via RL and consolidated from episodic experiences
- Supports **chunking** — sequences of actions that fire as a unit

#### 3d. Spatial Memory
- Dynamic knowledge graph of spatial relationships (from RoboMemory architecture)
- Updated incrementally with local conflict detection

#### 3e. DiffMemory (Differentiable Memory)
- Lightweight MLP-based memory that learns compressed patterns from episodic experiences
- 2-layer MemoryMLP (384→1536→384) with GELU + LayerNorm + residual
- Patterns stored via gradient descent during offline consolidation step 3b
- Surprise-gated: only stores novel patterns (reconstruction error above threshold)
- Retrieval via forward pass + cosine similarity — no gradients during inference
- Bounded capacity (200 patterns) with LRU eviction
- Complements EpisodicMemory: Episodic stores exact experiences, DiffMemory stores generalized patterns

### 4. Learning Engine

#### 4a. RL Core
- **TD(λ) learning** with eligibility traces for multi-timescale credit assignment
- Reward prediction error (RPE) drives all learning:
  ```
  δ = R + γ·V(s') - V(s)
  ```
- The RPE signal is broadcast to multiple subsystems (value, policy, representation, social features)

#### 4b. Efficient Coding Objective
The combined objective from the 2025 *Nature Communications* paper:

```
L_total = L_task + λ · Complexity(z)
```

Where `Complexity(z)` is measured as mutual information between input and latent code, or as the L0 norm of latent activations.

This forces the model to find the simplest representation that still performs the task — which is exactly what enables **generalization**.

#### 4c. Social Feature Learning (SFL)

Based on the *Nature Human Behaviour* paper:

- Social features `f ∈ {majority_opinion, expert_endorsement, popularity, ...}` are encoded alongside non-social features
- The same RL mechanism learns **associations between social features and reward**:
  ```
  Q(f) ← Q(f) + α · (R - Q(f))
  ```
- The agent learns *which* social features to trust based on past experience
- Over time, this produces behavior that looks like fixed heuristics (copy majority, copy successful) but is actually learned and flexible

#### 4d. Bayesian World Model Inference

From the *OpenReview 2026* paper on language-guided learning:

- Maintains a structured, program-like world model (causal graph + transition dynamics)
- Both **experience** (state transitions) and **language** (advice, instructions) are treated as evidence
- Bayesian update:
  ```
  P(model | experience, language) ∝ P(experience | model) · P(language | model) · P(model)
  ```
- Language is interpreted through a **speaker model** — an LM that estimates how likely a human with given beliefs would produce a particular utterance
- This allows the agent to learn from both doing and being told

### 5. Metacognitive Controller

**The key differentiator from standard RL agents.**

Monitors and controls:
- **Uncertainty** — entropy of the policy or value distribution
- **Confidence** — calibration of probability estimates vs actual outcomes
- **Learning progress** — rate of improvement in reward over recent episodes
- **Strategy selection** — when to use model-free vs model-based, when to explore, when to ask for help

Implemented as a separate small network that reads features from other modules and outputs **metacognitive actions**:
- `[STOP_AND_THINK]` — pause execution, enter model-based planning
- `[ASK_FOR_HELP]` — request linguistic guidance (social learning)
- `[REPLAY]` — trigger offline consolidation
- `[EXPLORE]` — increase exploration noise
- `[SWITCH_STRATEGY]` — change between model-free and model-based

### 6. Language Interface

- Encoder-decoder transformer for natural language
- Serves both **input** (understanding instructions, advice, questions) and **output** (explaining reasoning, giving advice to others)
- Language is treated as evidence for the Bayesian world model (Section 4d)
- Enables **self-talk** — the agent can generate language to itself as a form of metacognitive reasoning (e.g., "I'm uncertain about this step, let me check my knowledge")

### 7. Action Selection (Dual-System)

Implements the brain's two decision systems:

| System | Speed | Flexibility | When Used |
|--------|-------|-------------|-----------|
| **Model-Free** | Fast (ms) | Rigid (habitual) | Routine situations, low uncertainty |
| **Model-Based** | Slow (s) | Flexible (planning) | Novel situations, high uncertainty |

- Model-free: direct Q-values → action (e.g., DQN or SAC)
- Model-based: use world model to simulate trajectories → select action via planning (e.g., MuZero or tree search)
- **Controller** decides which system to use based on metacognitive signals (uncertainty, task novelty, time pressure)

---

## Learning & Consolidation Cycle

```
┌────────────────────────────────────────────┐
│               ONLINE PHASE                  │
│  (real-time interaction)                    │
│                                            │
│  Observe → WM → Act → Get reward → Update  │
│  - Fast RL updates (TD error)              │
│  - WM buffer management                    │
│  - Episodic storage of salient events      │
└────────────────────┬───────────────────────┘
                     │  (triggered by metacognitive
                     │   controller or idle time)
                     ▼
┌────────────────────────────────────────────┐
│              OFFLINE PHASE                 │
│  (consolidation, "sleep")                  │
│                                            │
│  1. Replay: sample episodes from memory     │
│  2. Prioritize: high-RPE events first      │
│  3. Abstract: compress repeated patterns   │
│     into schemata (semantic memory)        │
│  4. Prune: remove redundant/noisy memories │
│  5. Update world model: Bayesian update    │
│  6. Update procedural policies: offline RL │
└────────────────────────────────────────────┘
```

---

## Key Novel Properties

1. **Self-supervised representation learning via efficient coding** — the model automatically finds compact, generalizable representations without manual feature engineering. This is what enables human-like generalization from few examples.

2. **Social learning as emergent behavior** — instead of hardcoded social strategies, the model learns which social cues predict reward through experience, reproducing the flexibility seen in humans (*Nat. Hum. Behav.*).

3. **Metacognition as a trainable skill** — the metacognitive controller learns to monitor and adjust its own learning process, analogous to how humans improve at self-regulated learning with practice.

4. **Language bridges experience and knowledge** — linguistic input is treated as probabilistic evidence, not hard constraints. A bad instruction can be overridden by contradictory experience (matching human behavior).

5. **Individual differences emerge from latent state representations** — two copies of the same architecture can develop different "personalities" based on how they partition experience into latent states, explaining the variability seen in human learning.

6. **Catastrophic forgetting resistance** — the multi-memory system + offline consolidation + efficient coding means the model can learn continually without overwriting previous knowledge.

7. **Sample efficiency** — the combination of model-based planning, prior knowledge integration, and efficient coding should dramatically reduce the data needed compared to standard deep RL.

---

## Implementation

The full implementation lives at `src/project_adam/` — 20 components across 25 source files, 193 tests (incl. 57 architecture compliance tests).

```
1. Sensory Encoders: β-VAE with learned prior, top-10% sparsity, hardware-tier-aware
2. Working Memory: 64-slot bounded buffer with attention-gated eviction + temporal decay
3. Episodic Memory: SQLite-backed (s,a,r,c) tuples with symbolic keyword index + temporal compression
4. Semantic Memory: Schema graph with prediction-error gated assimilation/accommodation
5. Procedural Memory: Skill + ChunkedSkill classes with RPE-driven Q-learning
6. Spatial Memory: 17-relation triple store with conflict detection + inverse inference
7. DiffMemory: 2-layer MemoryMLP trained during consolidation, fast-forward retrieval
8. RL Core: TD(λ) + ActorNetwork policy head with eligibility traces + RPE broadcast
9. SFL Module: 7 social features, Rescorla-Wagner Q-learning, compute_temperature()
10. Bayesian World Model: Conjugate Gaussian priors, causal graph, speaker model
11. WebSearch: DDGS general + Wikipedia knowledge, independent caches
12. Metacognitive Controller: 5→32→16→5 MLP with REINFORCE, 5 canonical actions
13. LanguageInterface: Dual-backend (local/API), persona builder, behavioral rules
14. ActionSelector: Dual-system — pattern-based fast path + world model slow path
15. OfflineConsolidator: Full 6-step cycle with RPE prioritization + DiffMemory update
16. SelfPlayLearner: Autonomous background Q→A generation, 4 strategies, metacog-gated
17. PersonaManager: List, load, switch, generate personas via teacher API
18. MCP Server: 13 tools for external AI integration
19. UserProfileManager: Per-user state, LoRA adapters, rule weights
20. Persona: Markdown-based identity overlay with behavioral rules
```

Backend: PyTorch on consumer GPU (4GB+ VRAM minimum, 8GB+ recommended) or remote API fallback.
