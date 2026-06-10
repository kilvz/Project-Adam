import threading
import time
import re

import numpy as np
from ..config import get_memory_dir
from .store import SQLiteStore

_COMPRESSION_SIM = 0.92


class EpisodicMemory:
    def __init__(self, embedder_name="all-MiniLM-L6-v2"):
        self.episodes = []
        self._symbolic_index = {}
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
        self._rebuild_symbolic_index()

    def _rebuild_symbolic_index(self):
        self._symbolic_index.clear()
        for i, ep in enumerate(self.episodes):
            text = ep.get("text", "")
            for word in re.findall(r'\b[a-z]{4,}\b', text.lower()):
                self._symbolic_index.setdefault(word, []).append(i)

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

    def add(self, text, reward=0.0, rpe=None, context=None, action=None,
            backend=None):
        with self._lock:
            if self.embedder and len(self.episodes) > 0:
                new_emb = self.encode(text)
                match_idx = None
                match_sim = _COMPRESSION_SIM
                for i, ep in enumerate(self.episodes):
                    ep_emb = ep.get("emb")
                    if ep_emb is not None and ep_emb.size > 0:
                        sim = float(new_emb @ ep_emb / (np.linalg.norm(new_emb) * np.linalg.norm(ep_emb) + 1e-8))
                        if sim > match_sim:
                            match_sim = sim
                            match_idx = i
                if match_idx is not None:
                    existing = self.episodes[match_idx]
                    n = existing.get("count", 1)
                    existing["reward"] = (existing["reward"] * n + reward) / (n + 1)
                    existing["count"] = n + 1
                    if rpe is not None:
                        existing["rpe"] = rpe
                    existing["ts"] = time.time()
                    if action is not None:
                        existing["action"] = action
                    if backend is not None:
                        existing["backend"] = backend
                    self.save()
                    return

            entry = {
                "text": text,
                "state": text,
                "action": action,
                "reward": reward,
                "rpe": rpe,
                "context": context,
                "backend": backend,
                "ts": time.time(),
                "count": 1,
            }
            if self.embedder:
                entry["emb"] = self.encode(text)
            self.episodes.append(entry)
            idx = len(self.episodes) - 1
            for word in re.findall(r'\b[a-z]{4,}\b', text.lower()):
                self._symbolic_index.setdefault(word, []).append(idx)
        self.save()

    def update_last_action(self, action_text, backend=None):
        with self._lock:
            if self.episodes:
                self.episodes[-1]["action"] = action_text
                if backend is not None:
                    self.episodes[-1]["backend"] = backend

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

    def search_by_keyword(self, keyword, k=5):
        keyword = keyword.lower().strip()
        with self._lock:
            indices = self._symbolic_index.get(keyword, [])
            if not indices:
                return []
            results = [(self.episodes[i]["text"],
                        self.episodes[i].get("reward", 0))
                       for i in indices[-k:]]
        return results

    def recent(self, n=10):
        with self._lock:
            return list(self.episodes[-n:])

    def prune(self, threshold=0.3):
        with self._lock:
            before = len(self.episodes)
            self.episodes = [
                e for e in self.episodes
                if e.get("reward", 0) * 0.6 + 0.4 > threshold
            ]
            if len(self.episodes) > 100:
                self.episodes = self.episodes[-100:]
            if len(self.episodes) != before:
                self._rebuild_symbolic_index()
