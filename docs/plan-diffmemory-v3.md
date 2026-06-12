# DiffMemory v3 — Paper-Aligned Upgrades

Based on Titans (arXiv:2501.00663) and MIRAS (arXiv:2504.13173).

---

## Upgrade 1: Gradient Magnitude Surprise

**Paper**: Titans §3.2, eq. 9-10 — "surprise metric = gradient norm"

**Current**: `surprise = ‖MLP(x) - x‖²` — reconstruction error, forward pass only.

**Target**: `surprise = ‖∇_θ ℓ(M_θ(k), v)‖` — L2 norm of gradients w.r.t. all MLP weights.

The paper defines the surprise metric as the gradient magnitude of the memory loss:
```
Surprise(k, v) = ‖∇_θ ℓ(M_θ(k), v)‖
```

This is more principled than reconstruction error because:
- A pattern with low reconstruction error can still have high gradient (MLP hasn't fully internalized it)
- Gradient magnitude directly measures "how much would this pattern change my weights"
- Matches the paper's test-time training formulation exactly

### Change in `store()`:

```python
# Before: reconstruction error (forward only)
with torch.no_grad():
    reconstructed = self.model(emb)
    surprise = F.mse_loss(reconstructed, emb).item()

# After: gradient magnitude (requires backward)
self.model.train()
self.optimizer.zero_grad()
reconstructed = self.model(emb)
loss = F.mse_loss(reconstructed, emb)
loss.backward()
surprise = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
               if p.grad is not None) ** 0.5
```

**VRAM**: +0 MB (gradients already exist after backward).
**Lines changed**: ~10 in `store()`.

---

## Upgrade 2: Huber Loss Attentional Bias

**Paper**: MIRAS §3.2 — "Going beyond MSE and dot-product objectives"

**Current**: MSE loss for all surprise computations.

**MIRAS finding**: The YAAD variant uses Huber loss, which is less sensitive to outliers. MSE amplifies outlier errors quadratically; Huber grows linearly past its threshold δ. This makes memory more robust to noisy or atypical inputs.

### Change:

```python
# Configurable loss function
_LOSS_FNS = {
    "mse": F.mse_loss,
    "huber": F.huber_loss,
}

class DiffMemory:
    def __init__(self, ..., loss_fn="mse"):
        self.loss_fn = _LOSS_FNS.get(loss_fn, F.mse_loss)
```

Huber loss adds a `delta` parameter (default 1.0). The MIRAS paper's YAAD variant explicitly uses this for outlier robustness.

**VRAM**: +0 MB.
**Lines changed**: ~5 in `__init__`, ~2 in `store()`.

---

## Upgrade 3: Adaptive Forgetting (Learned Weight Decay)

**Paper**: Titans §3.2, eq. 13 and MIRAS §3.3 "Retention Regularization"

**Current**: Fixed `weight_decay=1e-4` in AdamW.

**Paper approach**: Titans learns a per-token decay factor via a tiny MLP head (`to_decay_factor`). MIRAS reinterprets this as "retention regularization" — the gate that controls how much of the old memory to retain vs overwrite.

### Change:

```python
# Learned decay head (tiny MLP)
self.to_decay = nn.Sequential(
    nn.Linear(dim, 8),
    nn.GELU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),  # output in (0, 1)
)

# In store():
decay = self.to_decay(emb).item()  # per-input retention rate
# Paper eq. 13: θ' = θ - λ·θ + η·g  (λ = decay)
# AdamW already does θ = θ - η·g - λ·θ via weight_decay
```

The learned decay allows the memory to adapt its forgetting rate per input — high decay for noisy inputs, low decay for important patterns.

**VRAM**: ~+0.01 MB (tiny MLP, 384×8 + 8×1 = 3080 params).
**Lines added**: ~10.

---

## Upgrade 4: Needle-in-Haystack Benchmark

**Paper**: Titans §4.3 — "Extreme Long-Context Recall" on BABILong

**Current**: No benchmark for DiffMemory retrieval accuracy.

### Test:

```python
def test_needle_in_haystack():
    dm = DiffMemory(dim=384, max_patterns=200)
    rng = np.random.RandomState(42)
    
    # 99 haystack patterns
    haystack = rng.randn(99, 384).astype(np.float32)
    dm.store(haystack, texts=[f"h{i}" for i in range(99)])
    
    # 1 needle pattern
    needle = rng.randn(1, 384).astype(np.float32)
    dm.store(needle, texts=["THE_NEEDLE"])
    
    results = dm.retrieve(needle[0], k=5)
    assert "THE_NEEDLE" in [r[0] for r in results]
```

## Implementation Order

| # | Task | Paper reference | Complexity |
|---|------|----------------|------------|
| 1 | Gradient magnitude surprise | Titans eq. 9-10 | Low (~10 lines) |
| 2 | Huber loss option | MIRAS §3.2 YAAD | Low (~5 lines) |
| 3 | Adaptive forgetting head | Titans eq. 13, MIRAS §3.3 | Medium (~15 lines) |
| 4 | Needle-in-haystack test | Titans §4.3 | Low (~20 lines) |

## VRAM Budget

| Upgrade | VRAM | Total DiffMemory |
|---------|------|-----------------|
| Current v2 | ~3 MB | ~3 MB |
| Gradient magnitude | +0 MB | ~3 MB |
| Huber loss | +0 MB | ~3 MB |
| Adaptive forgetting | +0.01 MB | ~3.01 MB |
| **Total** | **~3.01 MB** | |
