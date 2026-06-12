"""DiffMemory — lightweight differentiable memory module.

Learns compressed patterns from episodic experiences during consolidation.
Retrieves matching patterns during inference via a simple forward pass.
"""

import logging
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class _MemoryMLP(nn.Module):
    """2-layer MLP that acts as differentiable fast-weight memory."""

    def __init__(self, dim=384, hidden_mult=4):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
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
    """

    def __init__(self, dim=384, hidden_mult=4, max_patterns=200,
                 surprise_threshold=0.15, lr=1e-3, device="cpu"):
        self.dim = dim
        self.max_patterns = max_patterns
        self.surprise_threshold = surprise_threshold
        self.lr = lr
        self.device = device

        self.model = _MemoryMLP(dim=dim, hidden_mult=hidden_mult).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Stored pattern library: list of (embedding, text_snippet, usage_count)
        self.patterns = []
        self._pattern_lock = threading.Lock()

    def store(self, embeddings, texts=None):
        """Store patterns via gradient descent on the memory MLP.

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
                error = F.mse_loss(reconstructed, emb).item()

            if error < self.surprise_threshold:
                continue  # pattern already known, skip

            # Novel pattern — gradient update step
            self.model.train()
            self.optimizer.zero_grad()
            reconstructed = self.model(emb)
            loss = F.mse_loss(reconstructed, emb)
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            # Track pattern
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
        against stored patterns.

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

        # Transform query through memory MLP
        with torch.no_grad():
            query_transformed = self.model(query_t.unsqueeze(0)).squeeze(0).cpu().numpy()

        # Compare against stored pattern embeddings
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
        """Prune low-usage patterns to stay within capacity."""
        self._prune()

    def _prune(self):
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
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.patterns = state.get("patterns", [])

    def stats(self):
        with self._pattern_lock:
            avg_usage = (sum(p["usage_count"] for p in self.patterns) /
                         max(len(self.patterns), 1))
            return {
                "num_patterns": len(self.patterns),
                "max_patterns": self.max_patterns,
                "avg_usage": round(avg_usage, 2),
            }
