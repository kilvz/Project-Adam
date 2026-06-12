"""DiffMemory — lightweight differentiable memory module.

Learns compressed patterns from episodic experiences during consolidation.
Retrieves matching patterns during inference via a simple forward pass.

Titans-aligned features:
- Momentum-based surprise tracking (carries surprise across sequential tokens)
- Weight decay forgetting (AdamW + re-encode pruning)
- Configurable memory depth
"""

import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class _MemoryMLP(nn.Module):
    """N-layer MLP that acts as differentiable fast-weight memory.

    Depth can be configured (default 2). Deeper memories have higher
    capacity for compressing patterns, as shown in the Titans paper.
    """

    def __init__(self, dim=384, hidden_mult=4, depth=2):
        super().__init__()
        hidden = int(dim * hidden_mult)
        layers = []
        for i in range(depth):
            in_dim = dim if i == 0 else hidden
            out_dim = dim if i == depth - 1 else hidden
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        residual = x
        out = self.net(x)
        return self.norm(out + residual) * (self.gamma + 1.0)


class DiffMemory:
    """Differentiable memory for COGNET.

    Stores compressed patterns by updating MLP weights via gradient descent
    during consolidation. Retrieves patterns via simple forward pass during
    inference. No gradient computation during inference.

    Titans-aligned features:
    - Momentum: running average of surprise carries across sequential tokens
    - Weight decay: AdamW + re-encode pruning for principled forgetting
    - Configurable depth: deeper MLP for higher capacity
    """

    def __init__(self, dim=384, hidden_mult=4, depth=2, max_patterns=200,
                 surprise_threshold=0.15, momentum_beta=0.9,
                 momentum_scale=0.5, weight_decay=1e-4, lr=1e-3,
                 device="cpu"):
        self.dim = dim
        self.max_patterns = max_patterns
        self.surprise_threshold = surprise_threshold
        self.momentum_beta = momentum_beta
        self.momentum_scale = momentum_scale
        self.lr = lr
        self.device = device
        self.depth = depth

        self.model = _MemoryMLP(dim=dim, hidden_mult=hidden_mult,
                                depth=depth).to(device)
        # AdamW applies decoupled weight decay for principled forgetting
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Momentum buffer for surprise tracking across sequential tokens
        self._momentum = 0.0

        # Pattern text cache (for retrieval display).
        # Memory capacity is managed by MLP weights, not this cache.
        # Periodically pruned via re-encode: if MLP can't reconstruct a
        # pattern's embedding, it's been "forgotten" — drop it from cache.
        self.patterns = []
        self._pattern_lock = threading.Lock()

    def store(self, embeddings, texts=None):
        """Store patterns via gradient descent on the memory MLP.

        Uses momentum-based surprise tracking — if a token is surprising,
        the momentum carries over to subsequent tokens so related context
        is also stored. Weight decay in AdamW naturally forgets unused
        weight associations over time.

        Args:
            embeddings: np.ndarray or torch.Tensor, shape (N, dim).
            texts: Optional list of text snippets for retrieval.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float().to(self.device)
        else:
            embeddings = embeddings.float().to(self.device)

        for i in range(embeddings.shape[0]):
            emb = embeddings[i:i+1]
            with torch.no_grad():
                reconstructed = self.model(emb)
                surprise = F.mse_loss(reconstructed, emb).item()

            # Momentum: carry surprise across sequential tokens
            self._momentum = (
                self.momentum_beta * self._momentum
                + (1 - self.momentum_beta) * surprise
            )
            effective_surprise = surprise + self.momentum_scale * self._momentum

            if effective_surprise < self.surprise_threshold:
                continue

            # Novel pattern — gradient update step
            # AdamW applies weight decay automatically, decaying unused
            # weight associations (principled forgetting)
            self.model.train()
            self.optimizer.zero_grad()
            reconstructed = self.model(emb)
            loss = F.mse_loss(reconstructed, emb)
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            # Cache pattern text for retrieval display
            text = texts[i] if texts and i < len(texts) else ""
            with self._pattern_lock:
                self.patterns.append({
                    "embedding": emb.detach().cpu().numpy().flatten(),
                    "text": text,
                    "usage_count": 0,
                })

        self._prune()

    def retrieve(self, query_emb, k=3):
        """Retrieve top-k matching patterns.

        Forward pass through memory MLP, then nearest-neighbor match
        against cached patterns.

        Args:
            query_emb: np.ndarray or torch.Tensor, shape (dim,).
            k: Number of patterns to return.

        Returns:
            List of (text, similarity) tuples.
        """
        if not self.patterns:
            return []

        if isinstance(query_emb, np.ndarray):
            query_t = torch.from_numpy(query_emb).float().to(self.device)
        else:
            query_t = query_emb.float().to(self.device)

        with torch.no_grad():
            query_transformed = self.model(query_t.unsqueeze(0)).squeeze(0).cpu().numpy()

        scores = []
        for p in self.patterns:
            sim = float(np.dot(query_transformed, p["embedding"]) / (
                np.linalg.norm(query_transformed) * np.linalg.norm(p["embedding"]) + 1e-8
            ))
            scores.append((sim, p))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, p in scores[:k]:
            if sim > 0.5:
                p["usage_count"] += 1
                results.append((p.get("text", ""), round(sim, 3)))
        return results

    def consolidate(self):
        """Periodic maintenance — re-encode prune + momentum reset."""
        self._reencode_prune()
        self._momentum = 0.0

    def _reencode_prune(self):
        """Drop patterns the MLP has 'forgotten' (high reconstruction error).

        Re-encodes all cached pattern embeddings through the MLP. If the
        reconstruction error exceeds threshold, the MLP no longer represents
        that pattern — it's been naturally forgotten via weight decay.
        """
        with self._pattern_lock:
            if not self.patterns:
                return
            kept = []
            for p in self.patterns:
                emb_t = torch.from_numpy(p["embedding"]).float().to(self.device)
                with torch.no_grad():
                    reconstructed = self.model(emb_t.unsqueeze(0))
                    error = F.mse_loss(reconstructed, emb_t.unsqueeze(0)).item()
                if error < self.surprise_threshold * 2:
                    kept.append(p)
            dropped = len(self.patterns) - len(kept)
            if dropped:
                logger.debug("[diffmemory] Re-encode prune: dropped %d forgotten patterns", dropped)
            self.patterns = kept[-self.max_patterns:] if len(kept) > self.max_patterns else kept

    def _prune(self):
        """Simple capacity cap — keeps within max_patterns."""
        with self._pattern_lock:
            if len(self.patterns) <= self.max_patterns:
                return
            self.patterns.sort(key=lambda p: p["usage_count"])
            self.patterns = self.patterns[-self.max_patterns:]

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "patterns": self.patterns,
            "momentum": self._momentum,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.patterns = state.get("patterns", [])
        self._momentum = state.get("momentum", 0.0)

    def stats(self):
        with self._pattern_lock:
            avg_usage = (sum(p["usage_count"] for p in self.patterns) /
                         max(len(self.patterns), 1))
            return {
                "num_patterns": len(self.patterns),
                "max_patterns": self.max_patterns,
                "avg_usage": round(avg_usage, 2),
                "depth": self.depth,
                "momentum": round(self._momentum, 4),
            }
