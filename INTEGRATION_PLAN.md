# COGNET Integration Plan — Project Adam

## Architecture Gap Analysis

| COGNET Component | Implementation (adam_chat.py) | Status | Notes |
|---|---|---|---|
| **Sensory Encoders** | SensoryEncoder (line 266) | ✅ | VAE with KL loss, 896→128 dim compression |
| **Working Memory** | WorkingMemory (line 287) | ✅ | Bounded 8-turn gated buffer |
| **Episodic Memory** | EpisodicMemory (line 307) | ✅ | Vector store with sentence embeddings + reward tracking |
| **Semantic Memory** | SemanticMemory (line 356) | ✅ | Schema graph with assimilation/accommodation |
| **Procedural Memory** | NeuralMemory (line 403) | ✅ | Attention memory with online gradient updates (632K params) |
| **RL Core** | SFLModule (line 454) | ✅ | Q-learning Linear(4→1) over social features, per-turn SGD |
| **Social Feature Learning** | SFLModule + rule weights | ✅ | Q-values drive behavioral rule weighting per user |
| **Metacognitive Controller** | MetacognitiveController (line 478) | ✅ | Confidence/uncertainty estimation, strategy selection |
| **Language Interface** | Qwen2.5-0.5B-Instruct | ✅ | Frozen base model (494M params), instruction-tuned |
| **Action Selection** | ActionSelector (line 672) | ✅ | Dual-system: fast direct + slow web-backed when uncertain |
| **Offline Consolidation** | OfflineConsolidator (line 545) | ✅ | Background replay every 180s, schema extraction |
| **Web Search** | DuckDuckGo via WebSearch (line 522) | ✅ | External knowledge when metacognitive trigger fires |
| **Persona System** | Persona (line 21) | ✅ | Loads adam.md: 15 rules, 15 openings, 10 closings, 93 signatures |
| **Per-User Profiles** | UserProfileManager (line 180) | ✅ | Per-user state: topics, sentiment, rule weights, adopted phrases |

## Per-Turn Learning Pipeline

```
User input → detect_user() → extract_facts() → SematicMemory.add()
         → NeuralMemory.learn() [inline, 3 gradient steps]
         → compute_implicit_reward() [sentiment + engagement]
         → SFLModule.update() [Q-learning on social features]
         → ActionSelector.select(user_profile) [weighted rules + user context]
         → _update_rule_weights() [reward-driven per-user shifts]
         → _update_user_profile() [topics, sentiment, adopted phrases]
```

Every turn trains NeuralMemory + SFL + rule weights. No background thread needed for learning — it's synchronous and immediate.

## Code Structure

```
adam_chat.py (908 lines)
├── Persona                    # Line  21 — adam.md loader + adaptive prompt builder
├── UserProfileManager         # Line 180 — per-user profiles, persistence
├── SensoryEncoder             # Line 266 — VAE efficient coding bottleneck
├── WorkingMemory              # Line 287 — bounded 8-turn gated buffer
├── EpisodicMemory             # Line 307 — vector store with embeddings + reward
├── SemanticMemory             # Line 356 — schema graph with assimilation/accommodation
├── NeuralMemory               # Line 403 — attention memory, online gradient updates
├── SFLModule                  # Line 454 — Q-learning over social features
├── MetacognitiveController    # Line 478 — confidence/uncertainty/strategy
├── WebSearch                  # Line 522 — DuckDuckGo external knowledge
├── OfflineConsolidator        # Line 545 — background replay + abstraction
├── ActionSelector             # Line 672 — fast direct + slow web-backed
├── CognitiveAgent             # Line 738 — orchestrator + CLI
└── Helper functions           # Line 605-665 — extract_facts, reward, topics, detect_user
```
