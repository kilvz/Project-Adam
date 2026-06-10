import threading
import time

import numpy as np
from ..config import get_memory_dir
from .store import SQLiteStore


class EpisodicMemory:
    def __init__(self, embedder_name="all-MiniLM-L6-v2"):
        self.episodes = []
        self._lock = threading.Lock()
        self.embedder = None
        self._embedder_name = embedder_name
        self._init_embedder()
        self._store = SQLiteStore(
            "episodic_memory",
            path=get_memory_dir() / "episodic.pkl",
            pickle_fallback=lambda: [],
        )
        self.episodes = self._store.load(default=[])

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self._embedder_name)
        except Exception:
            pass

    def encode(self, text):
        if self.embedder is None:
            return np.zeros(384, dtype=np.float32)
        return self.embedder.encode(text, convert_to_numpy=True)

    def save(self):
        self._store.save(self.episodes)

    def add(self, text, reward=0.0):
        with self._lock:
            if self.embedder:
                emb = self.encode(text)
                self.episodes.append({
                    "text": text,
                    "emb": emb,
                    "reward": reward,
                    "ts": time.time(),
                })
            else:
                self.episodes.append({
                    "text": text,
                    "reward": reward,
                    "ts": time.time(),
                })
        self.save()

    def search(self, query, k=5):
        if not self.episodes or not query.strip():
            return []
        q_emb = self.encode(query)
        with self._lock:
            scored = []
            for ep in self.episodes:
                ep_emb = ep.get("emb")
                if ep_emb is None or ep_emb.size == 0:
                    continue
                sim = float(q_emb @ ep_emb / (np.linalg.norm(q_emb) * np.linalg.norm(ep_emb) + 1e-8))
                scored.append((ep["text"], sim, ep.get("reward", 0)))
            scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def recent(self, n=10):
        with self._lock:
            return list(self.episodes[-n:])

    def prune(self, threshold=0.3):
        with self._lock:
            self.episodes = [
                e for e in self.episodes
                if e.get("reward", 0) * 0.6 + 0.4 > threshold
            ]
            if len(self.episodes) > 100:
                self.episodes = self.episodes[-100:]
