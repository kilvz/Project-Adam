import numpy as np
from collections import defaultdict


class OfflineConsolidator:
    def __init__(self, episodic_memory, semantic_memory, world_model=None,
                 embedder=None, td_core=None, procedural_memory=None):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.world_model = world_model
        self.embedder = embedder
        self.td_core = td_core
        self.procedural = procedural_memory
        self.user_profiles = None

    def _distill_cross_user_patterns(self):
        if self.user_profiles is None:
            return
        all_profiles = self.user_profiles.profiles
        all_keywords = defaultdict(set)
        for name, profile in all_profiles.items():
            topics = profile.get("topics", {})
            for t in topics:
                all_keywords[t].add(name)
        for keyword, users in all_keywords.items():
            if len(users) >= 2:
                msg = f"[distilled] '{keyword}' mentioned by {', '.join(users)}"
                self.episodic.add(msg, reward=0.6)

    def _cluster_patterns(self, episodes):
        if self.embedder is None:
            return
        texts = [e.get("text", "") for e in episodes[-20:]
                 if len(e.get("text", "")) > 10]
        if len(texts) < 3:
            return
        try:
            embs = self.embedder.encode(texts, convert_to_numpy=True)
            sims = embs @ embs.T
            clusters = []
            used = set()
            for i in range(len(texts)):
                if i in used:
                    continue
                group = [texts[i]]
                used.add(i)
                for j in range(i + 1, len(texts)):
                    if j not in used and sims[i, j] > 0.7:
                        group.append(texts[j])
                        used.add(j)
                clusters.append(group[0])
            for phrase in clusters[:5]:
                self.episodic.add(f"[pattern] {phrase[:100]}", reward=0.5)
        except Exception:
            pass

    def _update_world_model(self, episodes):
        if self.world_model is None:
            return
        high_reward = [e for e in episodes if e.get("reward", 0) > 0.3]
        for ep in high_reward[-10:]:
            text = ep.get("text", "")
            if len(text) > 10:
                self.world_model.observe_from_text(text, ep.get("reward", 0))

    def _td_replay(self, episodes):
        if self.td_core is None:
            return
        replay = [e for e in episodes if e.get("rpe", 0) is not None]
        replay.sort(key=lambda e: abs(e.get("rpe", 0)), reverse=True)
        for ep in replay[:5]:
            text = ep.get("text", "")
            reward = ep.get("reward", 0)
            features = [0.0] * 7
            self.td_core.update(reward, features)

    def _update_procedural(self, episodes):
        if self.procedural is None:
            return
        high_reward = [e for e in episodes if e.get("reward", 0) > 0.3]
        for ep in high_reward[:10]:
            text = ep.get("text", "")
            action = ep.get("action", "") or ""
            if len(text) > 10 and action:
                self.procedural.record(text[:200], action[:200], ep.get("reward", 0))

    def merge_episodes(self, rpe=None):
        episodes = list(self.episodic.episodes)
        if not episodes:
            return

        weight_fn = lambda e: abs(e.get("rpe", e.get("reward", 0))) if rpe is None else abs(rpe)
        candidates = sorted(episodes, key=weight_fn, reverse=True)
        top = candidates[:10]

        for ep in top:
            text = ep.get("text", "")
            if len(text) < 10:
                continue
            if not any(w in text.lower() for w in ["my", "i", "you", "is", "are"]):
                continue
            self.semantic.add("experience", text[:200])
            self.episodic.add(f"[consolidated] {text[:100]}",
                              reward=ep.get("reward", 0) * 0.8)

        self.semantic.consolidate()
        self._distill_cross_user_patterns()
        self._cluster_patterns(episodes)
        self._td_replay(episodes)
        self._update_world_model(episodes)
        self._update_procedural(episodes)
        self.episodic.prune(threshold=0.3)

        if len(episodes) >= 5:
            recent = episodes[-5:]
            texts = [e.get("text", "") for e in recent if len(e.get("text", "")) > 20]
            if texts:
                merged = " ".join(texts[:3])
                self.semantic.add("merged_experience", merged[:300])
                self.episodic.add(f"[merged] {merged[:100]}", reward=0.5)
