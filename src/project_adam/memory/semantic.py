import threading
import numpy as np
from collections import defaultdict
from ..config import get_memory_dir
from .. import utils as project_adam_utils
from .store import SQLiteStore


class SemanticMemory:
    def __init__(self):
        self.schemas = defaultdict(lambda: {
            "facts": [], "emb": None, "observed_count": 0
        })
        self._lock = threading.Lock()
        self.embedder = None
        self._init_embedder()
        self._store = SQLiteStore(
            "semantic_memory",
            path=get_memory_dir() / "semantic.pkl",
            pickle_fallback=lambda: defaultdict(lambda: {
                "facts": [], "emb": None, "observed_count": 0
            }),
        )
        raw = self._store.load(default=None)
        if raw is not None:
            self.schemas = defaultdict(lambda: {"facts": [], "emb": None, "observed_count": 0}, raw)

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass

    def save(self):
        self._store.save(dict(self.schemas))

    def add(self, category, fact):
        with self._lock:
            if category not in self.schemas:
                self.schemas[category] = {"facts": [], "emb": None, "observed_count": 0}
            schema = self.schemas[category]
            schema["facts"].append(fact)
            schema["observed_count"] += 1
            combined = " ".join(schema["facts"])
            if self.embedder:
                schema["emb"] = self.embedder.encode(combined, convert_to_numpy=True)
        self.save()

    def retrieve(self, query, k=3):
        if not query.strip() or self.embedder is None:
            return []
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        with self._lock:
            scored = []
            for cat, data in self.schemas.items():
                emb = data.get("emb")
                if emb is None:
                    continue
                sim = float(q_emb @ emb / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
                scored.append((cat, data["facts"], sim))
            scored.sort(key=lambda x: -x[2])
        return scored[:k]

    def consolidate(self):
        with self._lock:
            to_remove = []
            for cat, data in self.schemas.items():
                if data["observed_count"] < 2 and len(data["facts"]) < 3:
                    to_remove.append(cat)
            for cat in to_remove:
                del self.schemas[cat]

    def extract_topics(self, text, embedder=None):
        embedder = embedder or self.embedder
        return project_adam_utils.extract_topics(text, embedder)

    def cross_user_distill(self, user_profiles):
        with self._lock:
            distilled = []
            all_keywords = defaultdict(set)
            for name, profile in user_profiles.items():
                topics = profile.get("topics", {})
                for t in topics:
                    all_keywords[t].add(name)
            for keyword, users in all_keywords.items():
                if len(users) >= 2:
                    distilled.append((keyword, list(users)))
            return distilled

    def phrase_cluster(self, phrases, threshold=0.7):
        if not phrases or self.embedder is None:
            return phrases[:10] if phrases else []
        embs = self.embedder.encode(phrases, convert_to_numpy=True)
        sims = embs @ embs.T
        clusters = []
        used = set()
        for i in range(len(phrases)):
            if i in used:
                continue
            group = [phrases[i]]
            used.add(i)
            for j in range(i + 1, len(phrases)):
                if j not in used and sims[i, j] > threshold:
                    group.append(phrases[j])
                    used.add(j)
            clusters.append(group[0])
        return clusters[:10]
