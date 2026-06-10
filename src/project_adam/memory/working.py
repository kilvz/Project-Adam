import time
import threading


class WorkingMemory:
    def __init__(self, max_turns=64, relevance_threshold=0.3, embedder=None,
                 episodic_memory=None, decay_tau=300.0):
        self.max_turns = max_turns
        self.relevance_threshold = relevance_threshold
        self.turns = []
        self.relevance_scores = []
        self._timestamps = []
        self._embedder = embedder
        self._episodic = episodic_memory
        self._lock = threading.Lock()
        self.goal = ""
        self.hypothesis_stack = []
        self._decay_tau = decay_tau

    def set_embedder(self, embedder):
        self._embedder = embedder

    def set_episodic_memory(self, episodic_memory):
        self._episodic = episodic_memory

    def set_goal(self, goal):
        with self._lock:
            self.goal = goal

    def push_hypothesis(self, hypothesis):
        with self._lock:
            self.hypothesis_stack.append(hypothesis)

    def pop_hypothesis(self):
        with self._lock:
            if self.hypothesis_stack:
                return self.hypothesis_stack.pop()
        return None

    def get_hypothesis(self):
        with self._lock:
            return self.hypothesis_stack[-1] if self.hypothesis_stack else None

    def _compute_relevance(self, content):
        if self._embedder is None or not self.turns:
            return 1.0
        try:
            emb = self._embedder.encode([content], convert_to_numpy=True)[0]
            context_texts = [t["content"] for t in self.turns[-4:]]
            context_embs = self._embedder.encode(context_texts, convert_to_numpy=True)
            sims = context_embs @ emb
            return float(max(sims))
        except Exception:
            return 1.0

    def add(self, role, content):
        now = time.time()
        relevance = self._compute_relevance(content)
        if relevance < self.relevance_threshold and self._episodic is not None:
            self._episodic.add(f"[wm-gate] {role}: {content[:200]}", reward=0.1)
            return
        with self._lock:
            self.turns.append({"role": role, "content": content})
            self.relevance_scores.append(relevance)
            self._timestamps.append(now)
            if len(self.turns) > self.max_turns:
                decayed = []
                for i in range(len(self.relevance_scores)):
                    age = now - self._timestamps[i]
                    decay = max(0.1, 1.0 - age / self._decay_tau)
                    decayed.append(self.relevance_scores[i] * decay)
                min_idx = min(range(len(decayed)), key=lambda i: decayed[i])
                evicted = self.turns.pop(min_idx)
                self.relevance_scores.pop(min_idx)
                self._timestamps.pop(min_idx)
                if self._episodic is not None:
                    self._episodic.add(f"[wm-evict] {evicted['role']}: {evicted['content'][:200]}", reward=0.2)

    def get_context(self, n=None):
        with self._lock:
            ctx = list(self.turns[-(n or self.max_turns):])
            if self.goal:
                ctx.insert(0, {"role": "goal", "content": self.goal})
            if self.hypothesis_stack:
                ctx.insert(0, {"role": "hypothesis", "content": self.hypothesis_stack[-1]})
            return ctx

    def clear(self):
        with self._lock:
            self.turns = []
            self.relevance_scores = []

    def get_gated_context(self, threshold=None):
        t = threshold or self.relevance_threshold
        with self._lock:
            return [turn for turn, score in zip(self.turns, self.relevance_scores)
                    if score >= t]
