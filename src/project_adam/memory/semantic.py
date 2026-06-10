import re
import threading
import numpy as np
from collections import defaultdict
from ..config import get_memory_dir
from .store import SQLiteStore


_SCHEMA_ASSIMILATION_THRESHOLD = 0.75
_SCHEMA_SPLIT_THRESHOLD = 0.8

_FACT_PATTERNS = [
    ("name", r"my name is (\w+)"),
    ("name", r"i(?:'m| am) called (\w+)"),
    ("location", r"i live in (\w[\w\s]*)"),
    ("location", r"i(?:'m| am) from (\w[\w\s]*)"),
    ("age", r"i(?:'m| am) (\d+) years? old"),
    ("preference", r"my favorite (\w+) is ([\w\s]+)"),
    ("likes", r"i like ([\w\s]+)"),
    ("dislikes", r"i don't? like ([\w\s]+)"),
]

_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "it", "its",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "this", "that", "to", "of", "in", "for", "on", "and", "or", "but",
    "not", "do", "does", "did", "have", "has", "had", "be", "been",
    "with", "about", "at", "by", "from", "as", "so", "if", "then",
    "what", "why", "how", "when", "where", "who", "which", "all",
    "can", "will", "would", "should", "could"}


class SemanticMemory:
    def __init__(self):
        self.schemas = {}
        self.graph = defaultdict(list)
        self._edges = []
        self._next_id = 0
        self._lock = threading.Lock()
        self.embedder = None
        self._init_embedder()
        self._store = SQLiteStore(
            "semantic_memory",
            path=get_memory_dir() / "semantic.pkl",
            pickle_fallback=lambda: ({}),
        )
        raw = self._store.load(default=None)
        if raw is not None:
            if isinstance(raw, tuple) and len(raw) == 4:
                self.schemas, self.graph, self._edges, self._next_id = raw
            elif isinstance(raw, tuple) and len(raw) == 3:
                self.schemas, self.graph, self._next_id = raw
            elif isinstance(raw, dict):
                if raw and all(k.isdigit() for k in raw):
                    self.schemas = raw
                    self._next_id = max(int(k) for k in raw) + 1
                else:
                    self.schemas = raw
                    self._next_id = len(raw)

    def _init_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass

    def save(self):
        self._store.save((self.schemas, dict(self.graph), self._edges, self._next_id))

    def _encode(self, text):
        if self.embedder is None:
            return None
        return self.embedder.encode(text, convert_to_numpy=True)

    def _cosine(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _find_best_schema(self, emb):
        best_id = None
        best_sim = 0.0
        for sid, schema in self.schemas.items():
            s_emb = schema.get("emb")
            sim = self._cosine(emb, s_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = sid
        return best_id, best_sim

    def _compute_prediction_error(self, emb, schema):
        s_emb = schema.get("emb")
        if s_emb is None:
            return 1.0
        return 1.0 - self._cosine(emb, s_emb)

    def _update_slots(self, schema, fact):
        slots = schema.setdefault("slots", {})
        words = fact.lower().split()
        for w in words:
            if len(w) > 3:
                slots[w] = slots.get(w, 0) + 1

    def _assimilate(self, schema_id, fact, emb, best_id=None, prediction_error=None):
        schema = self.schemas[schema_id]
        schema["prediction_error"] = prediction_error or self._compute_prediction_error(emb, schema)
        schema["facts"].append(fact)
        schema["observed_count"] += 1
        self._update_slots(schema, fact)
        if best_id is not None and best_id != schema_id:
            self.add_edge(schema_id, "assimilated_from", best_id)
        combined = " ".join(schema["facts"])
        schema["emb"] = self._encode(combined)
        return schema_id

    def _accommodate(self, category, fact, emb):
        sid = str(self._next_id)
        self._next_id += 1
        self.schemas[sid] = {
            "category": category,
            "facts": [fact],
            "emb": emb,
            "observed_count": 1,
            "prediction_error": 1.0,
            "slots": {},
        }
        self._update_slots(self.schemas[sid], fact)
        return sid

    def _check_split(self, schema_id):
        schema = self.schemas.get(schema_id)
        if not schema or len(schema["facts"]) < 3:
            return
        if self.embedder is None:
            return
        facts = schema["facts"]
        embs = self.embedder.encode(facts, convert_to_numpy=True)
        max_dist = 0.0
        pair = None
        n = len(facts)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine(embs[i], embs[j])
                if (1 - sim) > max_dist:
                    max_dist = 1 - sim
                    pair = (i, j)
        if max_dist > _SCHEMA_SPLIT_THRESHOLD and pair:
            i, j = pair
            self.schemas[schema_id]["facts"] = [facts[i]]
            self.schemas[schema_id]["emb"] = embs[i]
            self._accommodate(schema["category"], facts[j], embs[j])
            for k in range(n):
                if k != i and k != j:
                    sim_i = self._cosine(embs[k], embs[i])
                    sim_j = self._cosine(embs[k], embs[j])
                    if sim_i >= sim_j:
                        self.schemas[schema_id]["facts"].append(facts[k])
                    else:
                        new_id = str(self._next_id - 1)
                        self.schemas[new_id]["facts"].append(facts[k])
            combined_i = " ".join(self.schemas[schema_id]["facts"])
            self.schemas[schema_id]["emb"] = self._encode(combined_i)
            if self._next_id > 1:
                new_id = str(self._next_id - 1)
                combined_j = " ".join(self.schemas[new_id]["facts"])
                self.schemas[new_id]["emb"] = self._encode(combined_j)

    def add(self, category, fact):
        emb = self._encode(fact)
        with self._lock:
            best_id, best_sim = self._find_best_schema(emb)
            if best_id and best_sim >= _SCHEMA_ASSIMILATION_THRESHOLD:
                pred_err = 1.0 - best_sim
                sid = self._assimilate(best_id, fact, emb, best_id=best_id, prediction_error=pred_err)
                self._check_split(sid)
            else:
                sid = self._accommodate(category, fact, emb)
            self.graph[sid].append(category)
        self.save()
        return sid

    def add_edge(self, source_sid, relation, target_sid):
        with self._lock:
            if source_sid in self.schemas and target_sid in self.schemas:
                self._edges.append((source_sid, relation, target_sid))

    def get_related(self, sid, relation=None):
        with self._lock:
            results = []
            for s, r, t in self._edges:
                if s == sid and (relation is None or r == relation):
                    results.append((t, r, self.schemas.get(t, {}).get("category", "?")))
                if t == sid and (relation is None or r == relation):
                    results.append((s, r, self.schemas.get(s, {}).get("category", "?")))
            return results

    def traverse(self, sid, relation, max_hops=3):
        seen = set()
        frontier = [(sid, 0)]
        results = []
        with self._lock:
            while frontier:
                current, depth = frontier.pop(0)
                if current in seen or depth > max_hops:
                    continue
                seen.add(current)
                for s, r, t in self._edges:
                    if s == current and r == relation:
                        if t not in seen:
                            cat = self.schemas.get(t, {}).get("category", "?")
                            results.append((t, cat, depth + 1))
                            frontier.append((t, depth + 1))
        return results

    def retrieve(self, query, k=3):
        if not query.strip() or self.embedder is None:
            return []
        q_emb = self._encode(query)
        with self._lock:
            scored = []
            for sid, schema in self.schemas.items():
                sim = self._cosine(q_emb, schema.get("emb"))
                scored.append((schema.get("category", "unknown"), schema["facts"], sim))
            scored.sort(key=lambda x: -x[2])
        return scored[:k]

    def consolidate(self):
        with self._lock:
            to_remove = []
            for sid, schema in self.schemas.items():
                if schema["observed_count"] < 2 and len(schema["facts"]) < 3:
                    to_remove.append(sid)
            for sid in to_remove:
                del self.schemas[sid]
                self.graph.pop(sid, None)
                self._edges = [e for e in self._edges if e[0] != sid and e[1] != sid]

    def extract_facts(self, text):
        if not text or text.strip().endswith("?"):
            return []
        facts = []
        lower = text.lower()
        for cat, pattern in _FACT_PATTERNS:
            for m in re.finditer(pattern, lower):
                facts.append((cat, m.group(0).strip()))
        return facts

    def extract_topics(self, text, embedder=None):
        embedder = embedder or self.embedder
        words = list(set(w.strip(".,!?") for w in text.lower().split()
                         if w.strip(".,!?") not in _STOPWORDS and len(w) > 3))
        if embedder is not None and len(words) > 2:
            try:
                embs = embedder.encode(words, convert_to_numpy=True)
                sims = embs @ embs.T
                merged = []
                used = set()
                for i in range(len(words)):
                    if i in used:
                        continue
                    group = [words[i]]
                    used.add(i)
                    for j in range(i + 1, len(words)):
                        if j not in used and sims[i, j] > 0.65:
                            group.append(words[j])
                            used.add(j)
                    merged.append(group[0])
                return merged
            except Exception:
                pass
        return words

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
