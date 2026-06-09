# Training

Adam trains **per-user LoRA adapters** using reward as the curriculum signal.

## LoRA Adapters

Each user gets their own adapter at `agent_memory/adapters/{user}/adapter_model.safetensors`.

### Configuration

```
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

| Model | Trainable Params | VRAM |
|-------|-----------------|------|
| Qwen2.5-3B (NF4) | 1,843,200 | 2.1GB |
| Qwen2.5-1.5B (NF4) | 1,089,536 | 1.4GB |

### When Training Happens

Training triggers every **10 interactions**. It samples 5 reward-sorted episodes from that user's history:
- Episodes sorted by reward descending
- Negative-reward episodes are skipped
- Each episode gets a gradient weight: `max(0.1, min(2.0, reward + 1.0))`

### Training Loop

1. Load base model (4-bit NF4, eval mode)
2. Load adapter for the specific user (state dict into `lora_A`/`lora_B` weight tensors)
3. Format each episode as a chat template: `<|im_start|>user\n{msg}\n<|im_end|>\n<|im_start|>assistant\n{reply}`
4. Compute cross-entropy loss on the assistant tokens only
5. Scale loss by reward weight
6. Clip gradients (norm 1.0), update LoRA weights
7. Save adapter back to disk

### API

```python
agent = CognitiveAgent()

# Tune for a specific user
agent._update_user_adapter("Alice", reward=0.8, text="user: Hi\nassistant: Hello")

# Force a training cycle (if 10 interactions accumulated)
agent._train_adapter("Alice")
```

## Reward-Driven Curriculum (Phase E.26)

Instead of uniform sampling, training uses reward-weighted curriculum:

```python
weight = max(0.1, min(2.0, reward + 1.0))
if reward <= 0:
    continue  # skip negative-reward episodes entirely
loss = criterion(logits, labels) * weight
```

This ensures:
- High-reward experiences reinforce strongly
- Neutral experiences still contribute
- Negative experiences are discarded

## SFL Module

Social Feature Learning shapes generation temperature:

```
Features: [sentiment, engagement, interaction_norm, topic_novelty]
     ↓
SGD update → Q-value → softmax → temperature
```

Higher Q → lower temperature → more deterministic, "confident" responses.
Lower Q → higher temperature → more exploratory responses.

```python
sfl = SFLModule()
sfl.update(features=[0.7, 0.5, 0.9, 0.3], reward=0.8)
temp = sfl.select_action()  # 0.4–0.9 range
```

## Knowledge Gap Detection

When the user asks a wh-question (what, why, how, etc.) and confidence is low, the system automatically:

1. Searches the web via DuckDuckGo
2. Falls back to Wikipedia API
3. Injects search results into the context
4. Generates a grounded reply

## Inline Learning

Every interaction runs a lightweight inline learning step:
- 3 gradient steps on Neural Memory
- VAE training on the first 64 tokens
- No scheduler — simple SGD at learning rate 1e-5
