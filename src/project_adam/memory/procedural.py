import threading
import re


_SKILL_WORDS = re.compile(r"\w+")
_CHUNK_MIN_COUNT = 2


class ProceduralMemory:
    def __init__(self, max_skills=50, alpha=0.1):
        self.skills = []
        self._chunks = []
        self._history = []
        self._lock = threading.Lock()
        self._max_skills = max_skills
        self._alpha = alpha
        self._last_skill_idx = None

    def record(self, context_text, action_text, reward):
        if reward < 0.3:
            return
        keywords = set(_SKILL_WORDS.findall(context_text.lower()))
        if not keywords:
            return
        with self._lock:
            self._history.append((keywords, action_text))

            for idx, existing in enumerate(self.skills):
                if existing["keywords"] == keywords:
                    existing["success_count"] += 1
                    existing["total_count"] += 1
                    existing["action"] = action_text
                    self._try_chunk(keywords, action_text)
                    self._last_skill_idx = idx
                    return
            self.skills.append({
                "keywords": keywords,
                "action": action_text,
                "success_count": 1,
                "total_count": 1,
                "q_value": 0.5,
            })
            if len(self.skills) > self._max_skills:
                self.skills.sort(key=lambda s: s.get("q_value", 0.5))
                self.skills = self.skills[-self._max_skills:]

    def update_from_rpe(self, rpe):
        if self._last_skill_idx is not None:
            with self._lock:
                skill = self.skills[self._last_skill_idx]
                q = skill.get("q_value", 0.5)
                skill["q_value"] = q + self._alpha * (rpe - q)
                skill["q_value"] = max(0.0, min(1.0, skill["q_value"]))
                self._last_skill_idx = None

    def _try_chunk(self, keywords, action_text):
        if len(self._history) < _CHUNK_MIN_COUNT:
            return
        seq = tuple(a for _, a in self._history[-_CHUNK_MIN_COUNT:])
        for chunk in self._chunks:
            if chunk["sequence"] == seq:
                chunk["count"] += 1
                chunk["trigger_keywords"] = chunk["trigger_keywords"] | keywords
                return
        self._chunks.append({
            "sequence": seq,
            "count": 1,
            "trigger_keywords": keywords,
        })

    def retrieve(self, context_text, min_overlap=2):
        context_words = set(_SKILL_WORDS.findall(context_text.lower()))
        if not context_words:
            return None
        with self._lock:
            best_chunk = None
            best_chunk_score = 0.0
            for chunk in self._chunks:
                if chunk["count"] < _CHUNK_MIN_COUNT:
                    continue
                overlap = len(context_words & chunk["trigger_keywords"])
                if overlap >= min_overlap and overlap > best_chunk_score:
                    best_chunk_score = overlap
                    best_chunk = " | ".join(chunk["sequence"])

            best = None
            best_score = best_chunk_score
            for skill in self.skills:
                overlap = len(context_words & skill["keywords"])
                if overlap >= min_overlap:
                    score = overlap * skill.get("q_value", 0.5)
                    if score > best_score:
                        best_score = score
                        best = skill["action"]

            if best_chunk and best_chunk_score >= best_score:
                return best_chunk
            return best

    def record_failure(self, context_text):
        keywords = set(_SKILL_WORDS.findall(context_text.lower()))
        if not keywords:
            return
        with self._lock:
            for skill in self.skills:
                if skill["keywords"] == keywords:
                    skill["total_count"] += 1
                    return
