# DiffMemory v3 — Titans/MIRAS Alignment Roadmap

## Current alignment: ~80%

Four improvements to close the gap with Google's Titans/MIRAS research.

---

## 1. Gradient Magnitude Surprise

**Current**: `surprise = ‖MLP(x) - x‖²` — reconstruction error, forward pass only.

**Target**: `surprise = ‖∂loss/∂θ‖` — L2 norm of gradients w.r.t. MLP weights.

This is the actual "surprise metric" defined in the Titans paper. Gradient magnitude is a theoretically cleaner metric: a pattern that the MLP can reconstruct but hasn't fully internalized (high gradient despite low reconstruction error) should still be stored.

### Implementation

```python
def store(self, embeddings, texts=None):
    for i in range(embeddings.shape[0]):
        emb = embeddings[i:i+1]
        self.model.train()
        self.optimizer.zero_grad()
        reconstructed = self.model(emb)
        loss = F.mse_loss(reconstructed, emb)
        loss.backward()
        
        # Surprise = gradient magnitude (L2 norm of all gradients)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        surprise = total_norm ** 0.5
        
        # Momentum
        self._momentum = (
            self.momentum_beta * self._momentum
            + (1 - self.momentum_beta) * surprise
        )
        effective_surprise = surprise + self.momentum_scale * self._momentum
        
        if effective_surprise >= self.surprise_threshold:
            self.optimizer.step()
            # cache pattern text
        else:
            self.optimizer.zero_grad()  # discard gradients
```

**VRAM**: +0 MB (gradients already exist after backward).
**Cost**: One backward pass per token (already happens on store — moved earlier).

---

## 2. Deep Memory Experiment

**Current**: Default depth=2 layers.

**Titans finding**: Deeper memory → better perplexity, better scaling to long sequences.

**Action**: Run consolidation on the same episode batch with depth=2, 4, 6. Measure average reconstruction error after storage. If deeper = better (like Titans claims), change default to 4.

### Test script

```python
from project_adam.memory.diffmemory import DiffMemory
import numpy as np, time

embs = np.random.randn(100, 384).astype(np.float32)
texts = [f"pattern_{i}" for i in range(100)]

for depth in [2, 4, 6]:
    dm = DiffMemory(dim=384, depth=depth, surprise_threshold=0.01)
    t0 = time.time()
    dm.store(embs, texts=texts)
    dt = time.time() - t0
    # Re-encode and measure average error
    errors = []
    for emb in embs:
        emb_t = torch.from_numpy(emb).float()
        with torch.no_grad():
            recon = dm.model(emb_t.unsqueeze(0))
            errors.append(F.mse_loss(recon, emb_t.unsqueeze(0)).item())
    print(f"depth={depth}: avg_error={np.mean(errors):.4f}, time={dt:.2f}s")
```

**VRAM**: ~+3 MB per depth increase (2→4). GTX 1050 4GB can handle depth=6 (~9 MB).

---

## 3. MIRAS-Style Attentional Bias Options

**Current**: MSE loss for surprise computation only.

**MIRAS framework**: Generalizes beyond MSE to include Huber loss, L1, generalized norms. The YAAD variant uses Huber loss for outlier robustness.

**Action**: Add configurable loss function parameter to DiffMemory.

```python
_LOSS_FN = {
    "mse": F.mse_loss,
    "huber": F.huber_loss,
    "l1": F.l1_loss,
}

class DiffMemory:
    def __init__(self, ..., loss_fn="mse"):
        self.loss_fn = _LOSS_FN.get(loss_fn, F.mse_loss)
```

Huber loss is less sensitive to outliers — if a single embedding dimension has a large error (outlier), MSE amplifies it quadratically, while Huber grows linearly past its threshold. This could prevent a single weird token from dominating the memory update.

**VRAM**: +0 MB.
**Cost**: Negligible.

---

## 4. Needle-in-Haystack Benchmark

**Goal**: Validate that DiffMemory actually retrieves stored patterns correctly.

### Test design

1. Store 100 patterns via `diffmemory.store()` — diverse embeddings, realistic texts
2. Store 1 "needle" pattern with a unique text
3. Query with the needle's embedding
4. Assert the needle appears in top-3 retrieved results
5. Repeat with varying haystack sizes (50, 100, 200)

This is a simplified version of the BABILong benchmark used in the Titans paper.

### Implementation

```python
def test_needle_in_haystack():
    dm = DiffMemory(dim=384, max_patterns=200)
    rng = np.random.RandomState(42)
    
    # Haystack: 99 random patterns
    haystack = rng.randn(99, 384).astype(np.float32)
    haystack_texts = [f"haystack_{i}" for i in range(99)]
    dm.store(haystack, texts=haystack_texts)
    
    # Needle: 1 specific pattern
    needle = rng.randn(1, 384).astype(np.float32)
    dm.store(needle, texts=["THE_NEEDLE"])
    
    # Query with needle
    results = dm.retrieve(needle[0], k=5)
    texts = [r[0] for r in results]
    assert "THE_NEEDLE" in texts, "Needle not found in top-5!"
```

---

## Implementation Order

| # | Task | File | Est. |
|---|------|------|------|
| 1 | Gradient magnitude surprise | `memory/diffmemory.py` | 2h |
| 2 | Loss function options (MSE/Huber/L1) | `memory/diffmemory.py` + `config.py` | 1h |
| 3 | Deep memory experiment | Run script, no code change | 1h |
| 4 | Needle-in-haystack benchmark | `tests/test_diffmemory.py` | 3h |

Only tasks 1 and 2 require code changes. Tasks 3 and 4 are experiments/tests.

---

## VRAM Budget

| Task | VRAM added | Total DiffMemory VRAM |
|------|-----------|----------------------|
| Current v2 | — | ~3 MB (depth=2) |
| Gradient magnitude | +0 MB | ~3 MB |
| Loss function | +0 MB | ~3 MB |
| Deep memory (depth=6) | +6 MB | ~9 MB |
| **Worst case** | **+6 MB** | **~9 MB** (well within 4GB) |
