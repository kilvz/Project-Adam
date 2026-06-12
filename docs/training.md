# Training

Adam trains on every interaction through multiple learning signals.

## LoRA Adapters (Local Model)

Each user gets their own adapter at `agent_memory/adapters/{user}/adapter_model.safetensors`.

### Configuration

```python
LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```

| Model | Trainable Params |
|-------|-----------------|
| Qwen2.5-3B (NF4) | 1,843,200 |
| Qwen2.5-1.5B (NF4) | 1,089,536 |

### When Training Happens

Every **10 interactions**, `_lora_train_step()` samples 5 reward-sorted episodes:
- Episodes sorted by reward descending
- Negative-reward episodes are skipped
- Gradient weight: `max(0.1, min(2.0, reward + 1.0)) * rpe_scale`

### Online Distillation

When using `backend.mode: api`, the remote API generates replies that the local model trains on:
- API responses are stored in episodic memory with `backend: "api"` field
- LoRA training runs on ALL high-reward episodes regardless of backend
- Over time, the local model internalizes patterns from the remote API

## RL Core (TD(λ))

`TDCore` trains a value network and an actor network each turn:

```
δ = R + γ·V(s') - V(s)
RPE broadcast → SFL, encoder, metacog, procedural memory
```

- **ValueNetwork**: 8→64→1 MLP. Learned via TD(λ) with eligibility traces (`λ=0.8`)
- **ActorNetwork**: 8→64→5 MLP. Policy gradient with RPE.
- **Features**: sentiment, engagement, interaction_norm, topic_count, reward, sfl_q, enc_sparsity, sem_confidence

## SFL Module

7 social features learned via Rescorla-Wagner:

```
Features: [sentiment, engagement, interaction_norm, topic_novelty,
           majority_opinion, expert_endorsement, popularity]
Q(f) ← Q(f) + α · (RPE - Q(f))
temperature ← 0.7 - Q · 0.2 → [0.4, 0.9]
```

Higher Q → lower temperature → more deterministic responses.

## Sensory Encoder (β-VAE)

`SensoryEncoder` trains each turn on the user input embedding:
- Learned prior (Gaussian, `prior_mu`/`prior_logvar` parameters)
- Top-10% sparsity on latent code `z` (information bottleneck)
- Loss: `L_total = L_task + λ · Complexity(z)`
- Task loss weighted by metacognitive confidence
- Latent `z` stored in episodic memory for future retrieval

## Metacognitive Controller

`MetacogPolicy` (5→32→16→5 MLP) trained via REINFORCE:
- Features: [confidence, uncertainty, sfl_q, consecutive_low, learning_progress]
- Reward signal: same turn-level reward as all other components
- Mixes learned policy with rule-based baseline (epsilon-greedy)

## Reward Signal

Reward is computed from user input via `TDCore.compute_reward()`:
- Positive/negative word counting (sentiment lexicon)
- Optional embedding-based NLU refinement (cosine similarity to reference phrases)
- `reward = sentiment * 0.6 + engagement * 0.3`
- Reward range: [-1.0, 1.0]

## Consolidation Cycle

Every `REPLAY` meta-action triggers `OfflineConsolidator.merge_episodes()`:

1. **Replay + Prioritize** — sample episodes, sort by |RPE| descending, re-run TD on top 10
2. **Abstract** — cluster high-reward episodes by embedding similarity >0.7, record as procedural skills
3. **DiffMemory update** — encode high-reward episodes through differentiable memory MLP via gradient descent
4. **Prune** — remove low-reward episodes via `episodic.prune(threshold=0.2)`
5. **Update world model** — Bayesian update from high-reward episodes
6. **Update procedural** — offline RL from prioritized experiences, record as skills with Q-value update
7. **Bonus** — cross-user distillation, phrase clustering, TD core reset, semantic consolidation
