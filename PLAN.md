# Project Adam — Brain-Inspired Cognitive AI

## Goal
Build a conversational AI that learns permanently from interaction, powered by the COGNET architecture — a synthesis of 2025-2026 neuroscience and ML research (see `ARCHITECTURE.md`).

## Architecture (COGNET)

```
Metacognitive Controller ─┬─ Persona (adaptive behavioral rules)
                           ├─ Sensory Encoder (efficient coding VAE)
                           ├─ Working Memory (8-turn gated buffer)
                           ├─ Episodic Memory (vector store + reward)
                           ├─ Semantic Memory (schema graph)
                           ├─ Neural Memory (online gradient updates, 632K params)
                           ├─ SFL Module (social feature Q-learning, per-turn)
                           ├─ User Profiles (per-user topics/sentiment/weights)
                           ├─ Web Search (external knowledge)
                           ├─ Action Selector (fast/slow dual-system)
                           └─ Offline Consolidator (background replay)
```

All components run on a **4-bit Qwen2.5-1.5B-Instruct** base model (888M params at NF4, ~1.4GB VRAM) with LoRA fine-tuning (rank 8, 1M trainable params).

## Implementation Status

| Component | adam_chat.py | Status | Description |
|-----------|-------------|--------|-------------|
| Persona | Line 21 | ✅ | Loads adam.md: 15 if-then rules, 15 openings, 10 closings, 93 signatures |
| UserProfileManager | Line 180 | ✅ | Per-user state: topics, sentiment EMA, rule weights, adopted phrases |
| SensoryEncoder | Line 266 | ✅ | VAE bottleneck 896→128 dim with KL loss |
| WorkingMemory | Line 287 | ✅ | Bounded 8-turn gated context buffer |
| EpisodicMemory | Line 307 | ✅ | Vector store with sentence-BERT embeddings + reward |
| SemanticMemory | Line 356 | ✅ | Schema graph with assimilation/accommodation |
| NeuralMemory | Line 403 | ✅ | Attention memory (32 slots × 256 dim), online gradient updates |
| SFLModule | Line 454 | ✅ | Q-learning Linear(4→1), per-turn SGD over social features |
| WebSearch | Line 573 | ✅ | DDGS → Wikipedia fallback with search cache |
| OfflineConsolidator | Line 596 | ✅ | Prioritized replay, cross-user distillation, phrase clustering |
| ActionSelector | Line 837 | ✅ | Dual-system with SFL temperature + streaming output |
| CognitiveAgent | Line 917 | ✅ | Orchestrator: LoRA training, adapters, Gradio, dashboard |
| LoRA Adapters | - | ✅ | Per-user adapters, swapped on detection, 1M params each |
| Gradio Web UI | - | ✅ | `--web` flag → localhost:7860 ChatInterface |
| MetacognitiveController | Line 478 | ✅ | Confidence from logit entropy, strategy selection |
| WebSearch | Line 522 | ✅ | DuckDuckGo, triggered by metacognitive uncertainty |
| OfflineConsolidator | Line 545 | ✅ | Background replay every 180s, schema extraction |
| ActionSelector | Line 672 | ✅ | Dual-system: fast direct + slow web-backed |
| CognitiveAgent | Line 738 | ✅ | Full orchestrator + CLI with per-turn learning |

### Working Features
- [x] **Per-user identity** — auto-detects "my name is X", creates persistent profiles
- [x] **Per-turn NeuralMemory learning** — inline gradient update on every chat() call
- [x] **Social Feature Learning (SFL)** — Q-learning over sentiment/engagement/topic-novelty
- [x] **Behavioral rule adaptation** — per-user rule weights shift by reward signal
- [x] **Implicit reward** — sentiment analysis + engagement heuristic (no thumbs needed)
- [x] **Topic tracking** — extracts content words, builds per-user topic frequency map
- [x] **Phrase adoption** — tracks user trigrams, adopts at 5+ occurrences
- [x] **Weighted rule selection** — rules chosen by weight, not uniform random
- [x] **User-adapted system prompt** — includes name, interaction count, top topics, adopted phrases
- [x] Remembers names, locations, preferences across sessions
- [x] English conversation with instruction-tuned Qwen
- [x] Web search when uncertain (metacognitive trigger)
- [x] Background consolidation (extracts schemas, replays episodes)

### Current Limitations
- Small base model (0.5B) limits depth of reasoning
- Web search is keyword-triggered + metacognitive (not fully autonomous)
- Neural memory is tiny (32 slots × 256 dim) — proof of concept
- No persistent fine-tuning of base model (only memory system gradients)
- Sentiment analysis uses word lists (no NLU) — misses sarcasm/complex sentiment
- Phrase adoption is trigram-level (no semantic grouping)

## How to Run

```bash
cd "Project Adam"
python3 adam_chat.py
```

Commands inside chat:
- `/search <q>` — web search
- `/memory` — show episodic memory items
- `/schemas` — show semantic memory categories
- `/persona` — show loaded persona info
- `/profile` — show current user profile (topics, sentiment, weights)
- `/users` — list all known user profiles
- `/stats` — metacognitive stats (confidence, slow-path rate)
- `/save` — save all memory systems
- `/exit` — quit

## Future Work — Roadmap

### ✅ Phase A: Close the Feedback Loops (done)
Make existing components actually drive behavior — no new components, just connecting what exists:

1. **Close SFL → Action loop** — Q-values computed per turn but never consumed. Use Q-value as a "user satisfaction score" to gate which rules fire and whether to use opening/closing phrases.
2. **Mint new behavioral rules from patterns** — when a user mentions the same topic ≥5 times across sessions, mint a custom rule like `"If {topic} comes up, then {response from persona essence}"`. Store in `profile["custom_rules"]`, include in system prompt.
3. **Adopted phrases → active use** — inject top adopted phrases into system prompt as "this user says things like...", or append to closing phrases, so the model actually mirrors user language.
4. **Upgrade OfflineConsolidator** — add prioritized replay (high-reward first), abstraction (merge similar facts), pruning (remove low-utility). Aligns with architecture §7.
5. **NLU-based sentiment** — replace word-list sentiment with a small embedding classifier for more accurate reward signal (handles sarcasm, complex sentiment).

### ✅ Phase B: Deepen Architecture Alignment (done)
6. **Metacognitive action loop** — `MetacognitiveController.act()` emits `clarify` / `search` / `explore` / `replay` / `proceed`. Clarify returns a predefined question; explore lifts temperature to 0.85; replay triggers inline consolidation; search triggers web fetch.
7. **Connect SensoryEncoder to learning** — `SensoryEncoder.vae_loss()` called inside `_inline_learn()`. Compresses first 64 tokens through 896→128→896 VAE with recon + 0.001·KL loss, backprops into encoder weights.
8. **Cross-user pattern distillation** — consolidation scans all profiles, finds topics in ≥2 users, stores as `cross_user_topics` in semantic memory.
9. **Semantic phrase grouping** — consolidation clusters adopted phrases by embedding similarity (cosine > 0.7), stores clusters as `phrase_cluster` in semantic memory.
10. **Multi-session memory with importance-weighted retention** — replay priority uses `reward×0.6 + recency_factor×0.4`; pruning threshold applies to same importance metric.

### ✅ Phase C: Scale (done)
11. **4-bit Qwen2.5-1.5B via bitsandbytes** — 1.5B params at NF4 quantization = 1.4GB VRAM (vs 1.0GB for 0.5B at fp16). `BASE_MODEL` defaults to `MODEL_1_5B` with automatic fallback to `MODEL_0_5B` fp16. Full chat template, hidden_size=1536, 28 layers.
12. **LoRA fine-tuning from accumulated memories** — model wrapped with PEFT LoRA (`q_proj`, `v_proj`, rank=8) at init. Every 10 interactions, `_lora_train_step()` runs 5 gradient steps on recent episodes (2.3s overhead). Adapter saved to `agent_memory/adapters/default/` and reloaded at startup.
13. **Autonomous knowledge-gap detection** — when input is a factual question (wh-word + `?`) AND semantic retrieval confidence < 0.25 AND no facts extracted → auto-trigger web search → store results in semantic memory as `web_knowledge`. Self-educating without explicit `/search`.

## Final System Summary

| Component | Status | Details |
|-----------|--------|---------|
| Persona (adam.md) | ✅ | 15 rules, 15 openings, 10 closings, 93 signatures |
| SensoryEncoder (VAE) | ✅ | 896/1536→128 bottleneck, β=0.001 KL, trainable inline |
| WorkingMemory | ✅ | 8-turn gated buffer |
| EpisodicMemory | ✅ | Vector store + importance-weighted retention |
| SemanticMemory | ✅ | Schema graph with cross-user distillation + phrase clusters |
| NeuralMemory | ✅ | Transformer attention, 3 gradient steps/turn |
| SFL Module | ✅ | Q-learning (4 features), per-turn, <1ms |
| User Profiles | ✅ | Per-user topics, sentiment, rule weights, custom rules |
| Metacognitive Controller | ✅ | 5 actions: clarify/explore/search/replay/proceed |
| Action Selector | ✅ | Dual-system with SFL temperature + web search |
| Offline Consolidator | ✅ | Prioritized replay, pruning, dedup, cross-user, phrase clusters |
| Model | ✅ | 4-bit Qwen2.5-1.5B (1.4GB VRAM), auto-fallback to 0.5B |
| LoRA Fine-tuning | ✅ | Rank 8 adapters, trained every 10 interactions |
| Auto Knowledge-Gap | ✅ | Question + low confidence → auto-search |

### ✅ Phase D: Polish & UX
14. **Streaming output** — `TextIteratorStreamer` in a background thread with token-by-token `print()` or optional callback. First token appears in <1s on Pascal.
15. **Evaluation dashboard** — `/dashboard` command showing SFL Q, reward trend, confidence, rule weight spread, last action. ASCII box format.
16. **Per-user LoRA adapters** — Each user gets their own LoRA adapter saved to `agent_memory/adapters/{name}/`. Swapped via `_switch_adapter()` on user detection.
17. **Web search fallback** — Wikipedia API fallback when DDGS returns 0 results. Search cache persisted to `agent_memory/search_cache.json`. All tiers (DDGS → Wikipedia → cache) integrated into `WebSearch.search()`. Also added `.external/` config with web-search agent and commands.
18. **Improved topic extraction** — `extract_topics()` now optionally takes an embedder to merge semantically similar words (cosine > 0.65 → same topic). Uses `all-MiniLM-L6-v2`.
19. **Web UI (Gradio)** — `gradio.ChatInterface` at `localhost:7860` via `--web` flag. Token callback adapter for streaming. `pip install gradio` required.
20. **Qwen2.5-3B 4-bit** — *(not started)*
21. **Voice mode** — *(not started)*
